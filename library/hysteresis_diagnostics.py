"""Bounded summaries and aperture previews for hysteresis reconstructions."""
from collections import deque
import numpy as np


class HysteresisHistory:
    """Keep scalar/1D summaries only; never retain callback fields or components."""
    def __init__(self, holograms, mask_pixel, state_labels, material_mask,
                 max_steps=256, bins=80, value_range=(-2., 2.), max_samples=8192):
        if max_steps < 1 or bins < 2 or max_samples < 1:
            raise ValueError('Positive history limits and at least two bins required.')
        self.records = deque(maxlen=max_steps)
        self.edges = np.linspace(*value_range, bins+1)
        self.labels = list(state_labels)
        self.active = np.flatnonzero(material_mask)
        self.samples = []
        masks = np.broadcast_to(mask_pixel, np.shape(holograms))
        for image, mask in zip(holograms, masks):
            valid = np.flatnonzero((~np.asarray(mask, bool)) & np.isfinite(image) & (image >= 0))
            valid = valid[::max(1, int(np.ceil(len(valid)/max_samples)))]
            self.samples.append((valid, np.asarray(image).ravel()[valid].copy()))
        self.total_steps = 0

    def __call__(self, event):
        if event['stage'] not in ('physical_projection', 'final_physical_projection'):
            return
        fields, components = event['fields'], event['components']
        errors = []
        for field, (indices, target) in zip(fields, self.samples):
            modes = field[None] if field.ndim == 2 else field
            predicted = sum(abs(mode.ravel()[indices])**2 for mode in modes)
            norm = np.linalg.norm(target)
            errors.append(float(np.linalg.norm(predicted-target)/norm) if norm else np.nan)
        counts = np.zeros((len(self.labels), len(self.edges)-1), np.uint32)
        overflow = np.zeros((len(self.labels), 2), np.uint32)
        quantiles = np.full((len(self.labels), 3), np.nan, np.float32)
        means = np.full(len(self.labels), np.nan, np.float32)
        maps = components.get('magnetization')
        if maps is not None:
            order = [components['state_names'].index(s) for s in self.labels]
            for i, state in enumerate(order):
                plane = maps[state].ravel()
                total, number = 0., 0
                for start in range(0, len(self.active), 65536):
                    values = plane[self.active[start:start+65536]]
                    values = values[np.isfinite(values)]
                    counts[i] += np.histogram(values, self.edges)[0].astype(np.uint32)
                    overflow[i] += np.asarray([np.count_nonzero(values < self.edges[0]),
                                               np.count_nonzero(values > self.edges[-1])], np.uint32)
                    total += values.sum(); number += values.size
                sample = plane[self.active[::max(1, int(np.ceil(len(self.active)/4096)))]]
                sample = sample[np.isfinite(sample)]
                if sample.size:
                    quantiles[i] = np.percentile(sample, [10, 50, 90])
                if number:
                    means[i] = total/number
        factors = components.get('aperture_illumination_factors')
        if factors is None:
            factors = np.sqrt(np.asarray(components.get('illumination_intensity_factors',
                                                       np.ones(len(fields)))))[:, None]
            factors = factors*np.exp(1j*np.asarray(components.get('global_phase_offsets_rad',
                                                                 np.zeros(len(fields)))))[:, None]
        self.total_steps += 1
        self.records.append(dict(step=self.total_steps, stage=event['stage'],
            outer_round=event.get('outer_round', -1), detector_error=np.asarray(errors, np.float32),
            fit_error=float(components.get('fit_residual_rms', np.nan)),
            reference_error=float(components.get('reference_aperture_profile_residual_rms', np.nan)),
            factors=np.asarray(factors, np.complex64).copy(), histogram=counts,
            overflow=overflow, quantiles=quantiles, mean=means))
        record = self.records[-1]
        modal_maps = components.get('magnetization_by_mode_group_packed')
        if modal_maps is not None:
            group_counts=[];group_means=[];group_quantiles=[]
            for group in range(len(components['magnetization_mode_groups'])):
                packed=modal_maps[str(group)][...,0]
                rows=[]; averages=[]; ranges=[]
                for state in [components['state_names'].index(label) for label in self.labels]:
                    plane=packed[state]
                    histogram=np.zeros(len(self.edges)-1,np.uint32)
                    total=0.;number=0
                    for start in range(0,len(plane),65536):
                        values=plane[start:start+65536];values=values[np.isfinite(values)]
                        histogram+=np.histogram(values,self.edges)[0].astype(np.uint32)
                        total+=values.sum();number+=len(values)
                    sample=plane[::max(1,int(np.ceil(len(plane)/4096)))];sample=sample[np.isfinite(sample)]
                    rows.append(histogram);averages.append(total/number if number else np.nan)
                    ranges.append(np.percentile(sample,[10,50,90]) if len(sample) else [np.nan]*3)
                group_counts.append(rows);group_means.append(averages);group_quantiles.append(ranges)
            record['modal_histogram']=np.asarray(group_counts,np.uint32)
            record['modal_mean']=np.asarray(group_means,np.float32)
            record['modal_quantiles']=np.asarray(group_quantiles,np.float32)
            record['modal_group_labels']=np.asarray(['modes '+','.join(str(m+1) for m in group)
                                    for group in components['magnetization_mode_groups']])
        modal_corrections=components.get('illumination_corrections_by_mode')
        if modal_corrections:
            mode_ids=sorted(int(mode) for mode in modal_corrections)
            record['modal_factor_modes']=np.asarray(mode_ids)
            record['modal_factors']=np.asarray([modal_corrections[str(mode)].get('aperture_illumination_factors',
                np.sqrt(modal_corrections[str(mode)]['illumination_intensity_factors'])[:,None]
                *np.exp(1j*modal_corrections[str(mode)]['global_phase_offsets_rad'])[:,None])
                for mode in mode_ids],np.complex64)

    def export(self):
        result = dict(histogram_edges=self.edges, state_labels=self.labels,
                      total_steps=self.total_steps, max_stored_steps=self.records.maxlen,
                      detector_error_scope='Sampled unblurred detector intensity; excludes coherence kernel',
                      quantile_scope='At most 4096 OH pixels per state; histogram and mean use all OH pixels')
        if self.records:
            for key in self.records[0]:
                result[key] = np.asarray([record[key] for record in self.records])
        return result


def plot_history(history, folder):
    import matplotlib.pyplot as plt
    h = history.export() if isinstance(history, HysteresisHistory) else history
    if 'step' not in h:
        print('No recorded projection history; rerun the physical stage to collect it.')
        return
    x = h['step']; labels = h['state_labels']
    fig, axes = plt.subplots(1, 3, figsize=(15,4), layout='constrained')
    for i, label in enumerate(labels):
        axes[0].plot(x, h['detector_error'][:,i], '.-', label=label)
        axes[2].plot(x, h['mean'][:,i], '.-', label=label)
        axes[2].fill_between(x, h['quantiles'][:,i,0], h['quantiles'][:,i,2], alpha=.08)
    axes[0].set_title('Sampled unblurred detector intensity error')
    axes[1].plot(x, h['fit_error'], '.-', label='Physical log fit')
    axes[1].plot(x, h['reference_error'], '.-', label='RH profile mismatch before projection')
    axes[1].legend(fontsize=7); axes[1].set_title('Projection errors')
    axes[2].set_title('OH mean and sampled 10–90% range')
    for ax in axes: ax.set_xlabel('Physical projection step')
    axes[0].legend(fontsize=6, ncol=2)
    fig.savefig(folder/'step_errors_magnetization.png', dpi=150); plt.show(); plt.close(fig)
    factors = h['factors']
    fig, axes = plt.subplots(2, factors.shape[2], figsize=(3*factors.shape[2],6),
                             squeeze=False, layout='constrained')
    for aperture in range(factors.shape[2]):
        for state, label in enumerate(labels):
            axes[0,aperture].plot(x, abs(factors[:,state,aperture])**2, '.-', label=label)
            axes[1,aperture].plot(x, np.unwrap(np.angle(factors[:,state,aperture])), '.-')
        axes[0,aperture].set_title(f'Aperture {aperture+1}')
        axes[1,aperture].set_xlabel('Physical projection step')
    axes[0,0].set_ylabel('Intensity factor'); axes[1,0].set_ylabel('Factor phase (rad)')
    axes[0,0].legend(fontsize=6)
    fig.savefig(folder/'step_aperture_factors.png', dpi=150); plt.show(); plt.close(fig)
    columns = min(4, len(labels)); rows = (len(labels)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(4*columns,3*rows), squeeze=False, layout='constrained')
    for i, ax in enumerate(axes.flat):
        if i >= len(labels): ax.set_visible(False); continue
        counts = h['histogram'][:,i].astype(float)
        probability = counts/np.maximum(counts.sum(axis=1,keepdims=True),1)
        ax.pcolormesh(x, (h['histogram_edges'][:-1]+h['histogram_edges'][1:])/2,
                      probability.T, shading='nearest', cmap='magma', vmin=0)
        ax.set(title=labels[i], xlabel='Physical projection step', ylabel='OH magnetization')
    fig.savefig(folder/'step_oh_distributions.png', dpi=150); plt.show(); plt.close(fig)
    print('OH histogram counts outside plot range (low, high):', h['overflow'][-1].tolist())
    if 'modal_histogram' in h:
        for group,group_label in enumerate(h['modal_group_labels'][0]):
            fig,axes=plt.subplots(rows,columns,figsize=(4*columns,3*rows),squeeze=False,layout='constrained')
            for state,ax in enumerate(axes.flat):
                if state>=len(labels):ax.set_visible(False);continue
                counts=h['modal_histogram'][:,group,state].astype(float)
                probability=counts/np.maximum(counts.sum(axis=1,keepdims=True),1)
                ax.pcolormesh(x,(h['histogram_edges'][:-1]+h['histogram_edges'][1:])/2,
                              probability.T,shading='nearest',cmap='magma')
                ax.plot(x,h['modal_mean'][:,group,state],color='cyan',lw=1)
                ax.set(title=labels[state],xlabel='Physical projection step',ylabel='OH magnetization')
            fig.suptitle('Magnetic group '+str(group+1)+': '+group_label)
            fig.savefig(folder/f'step_oh_distribution_group_{group+1}.png',dpi=150);plt.show();plt.close(fig)
    if 'modal_factors' in h:
        for mode_index,mode in enumerate(h['modal_factor_modes'][0]):
            factors=h['modal_factors'][:,mode_index]
            fig,axes=plt.subplots(2,factors.shape[2],figsize=(3*factors.shape[2],6),squeeze=False,layout='constrained')
            for aperture in range(factors.shape[2]):
                for state,label in enumerate(labels):
                    axes[0,aperture].plot(x,abs(factors[:,state,aperture])**2,'.-',label=label)
                    axes[1,aperture].plot(x,np.unwrap(np.angle(factors[:,state,aperture])),'.-')
                axes[0,aperture].set_title(f'Aperture {aperture+1}')
                axes[1,aperture].set_xlabel('Physical projection step')
            axes[0,0].set_ylabel('Intensity factor');axes[1,0].set_ylabel('Factor phase (rad)')
            axes[0,0].legend(fontsize=6);fig.suptitle(f'Mode {mode+1}: aperture factors')
            fig.savefig(folder/f'step_aperture_factors_mode_{mode+1}.png',dpi=150);plt.show();plt.close(fig)



def plot_complex_apertures(objects, support, material, state_labels, folder,
                           states=(0,5,-1), norm=None, padding=.15, max_size=256, prefix=""):
    """Build RGB only for bounded aperture patches, never full images/stacks."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from scipy.ndimage import label, find_objects
    labels, _ = label(np.asarray(support) != 0)
    windows = []
    for number, slices in enumerate(find_objects(labels), 1):
        if slices is None: continue
        radius = int(np.ceil(max(s.stop-s.start for s in slices)*(1+2*padding)/2))
        center = [(s.start+s.stop)/2 for s in slices]
        window = tuple(slice(max(0,int(c-radius)), min(size,int(c+radius)))
                       for c,size in zip(center, labels.shape))
        step = max(1, int(np.ceil(max(s.stop-s.start for s in window)/max_size)))
        sampled = tuple(slice(s.start,s.stop,step) for s in window)
        kind = 'OH' if np.any(np.asarray(material)[labels==number]) else 'RH'
        windows.append((kind, number, sampled))
    windows.sort(key=lambda item: (item[0]!='OH', item[1]))
    selected = list(dict.fromkeys(i % len(objects) for i in states if -len(objects) <= i < len(objects)))
    if not windows or not selected: return
    if norm is None:
        samples = []
        for state in selected:
            for _,_,window in windows:
                patch = abs(objects[state][window]).ravel()
                samples.append(patch[::max(1,int(np.ceil(patch.size/4096)))])
        norm = max(float(np.percentile(np.concatenate(samples),99)),1e-30)
    if not np.isfinite(norm) or norm <= 0: raise ValueError('HSV norm must be finite and positive.')
    for state in selected:
        fig, axes = plt.subplots(1,len(windows),figsize=(3*len(windows),3),squeeze=False,layout='constrained')
        for ax,(kind,number,window) in zip(axes[0],windows):
            z = objects[state][window]
            hsv = np.empty((*z.shape,3),np.float32)
            hsv[...,0] = (np.angle(z)/(2*np.pi)+.5)%1
            hsv[...,1] = 1.; hsv[...,2] = np.clip(abs(z)/norm,0,1)
            ax.imshow(hsv_to_rgb(hsv),interpolation='nearest'); ax.set_title(f'{kind}: aperture {number}'); ax.axis('off')
        fig.suptitle(f'{prefix}{state_labels[state]} | phase hue, amplitude brightness | norm {norm:.3g}')
        fig.savefig(folder/f'{prefix}complex_apertures_state_{state}.png',dpi=150); plt.show(); plt.close(fig)


class _FocusedObjects:
    """Lazy display objects: cache only one observation's object plane."""
    def __init__(self, fields, recipe, energy_labels):
        self.fields, self.recipe, self.energy_labels = fields, recipe, energy_labels
        self.index = None
        self.object = None

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, index):
        if index != self.index:
            from . import phase_retrieval_universal as u
            focused = u._projection_focus_transform(self.fields[index:index+1],
                [self.energy_labels[index]], self.recipe.get('projection_focus_prop_um',0.),
                self.recipe.get('projection_focus_phase_rad',0.),self.recipe.get('projection_focus_setup'),
                integer_wavelength=self.recipe.get('projection_focus_integer_wavelength',True))[0]
            self.object = u.fourier_field_to_display_object(focused)
            self.index = index
        return self.object


def plot_equivalent_mode_diagnostics(fields, support, material, state_labels, recipe, components,
                                     folder, stage='final', states=(0,5,-1), norm=None, max_size=256):
    """Show each support-factor-one mode, processing one mode/plane at a time."""
    import matplotlib.pyplot as plt
    from . import phase_retrieval_geometry as geometry
    modes = np.asarray(fields)
    if modes.ndim == 3: modes = modes[:,None]
    factors = recipe.get('modes') or [1]*modes.shape[1]
    selected = [i for i,value in enumerate(factors) if value == 1]
    if len(selected)<2: return
    energies = [float(np.asarray(recipe['energy_values']).ravel()[0])]*len(modes)
    roi = geometry.support_bounding_roi(material,padding_fraction=.15)
    step = max(1,int(np.ceil(max(s.stop-s.start for s in roi)/max_size)))
    roi = tuple(slice(s.start,s.stop,step) for s in roi)
    columns=min(5,len(modes));rows=(len(modes)+columns-1)//columns
    def grid(images,title,filename,cmap,vmin,vmax):
        fig,axes=plt.subplots(rows,columns,figsize=(3*columns,3*rows),squeeze=False,layout='constrained')
        for state,ax in enumerate(axes.flat):
            if state>=len(images):ax.set_visible(False);continue
            im=ax.imshow(np.ma.masked_invalid(images[state]),cmap=cmap,vmin=vmin,vmax=vmax)
            ax.set_title(state_labels[state],fontsize=8);ax.axis('off')
        fig.suptitle(title)
        fig.colorbar(im,ax=[ax for ax in axes.flat if ax.get_visible()],shrink=.8)
        fig.savefig(folder/filename,dpi=150);plt.show();plt.close(fig)
    for mode in selected:
        objects=_FocusedObjects(modes[:,mode],recipe,energies)
        patches=np.stack([objects[state][roi] for state in range(len(modes))])
        prefix=f'{stage}_mode_{mode+1}_'
        title=f'{stage.capitalize()} mode {mode+1} (support factor 1)'
        limit=max(float(np.percentile(abs(patches),99.5)),1e-30)
        grid(abs(patches),title+': OH amplitude',prefix+'amplitude.png','gray',0,limit)
        grid(np.angle(patches),title+': OH phase',prefix+'phase.png','twilight',-np.pi,np.pi)
        del patches
        plot_complex_apertures(objects,support,material,state_labels,folder,states=states,
                              norm=norm,max_size=max_size,prefix=prefix)
        # Display the map linked to this mode, explicitly labeling shared groups.
        if components is not None and components.get('magnetization_mode_groups') is not None:
            group=next((g for g,indices in enumerate(components['magnetization_mode_groups']) if mode in indices),None)
            if group is not None:
                aperture=np.asarray(components['modal_physical_fit_aperture'])
                # Sample directly from the packed map without expanding state × full images.
                index_map=np.full(aperture.shape,-1,np.int32)
                index_map[aperture]=np.arange(aperture.sum())
                sampled=np.fft.fftshift(index_map)[roi]
                packed=components['magnetization_by_mode_group_packed'][str(group)][...,0]
                order=[components['state_names'].index(label) for label in state_labels]
                maps=np.stack([np.where(sampled>=0,packed[state,np.maximum(sampled,0)],np.nan) for state in order])
                members=[i+1 for i in components['magnetization_mode_groups'][group]]
                limit=max(1.,float(np.nanmax(abs(maps))))
                grid(maps,title+f': magnetization group {group+1}, modes {members}',
                     prefix+'magnetization.png','RdBu_r',-limit,limit)
        elif components is not None and mode==0:
            order=[components['state_names'].index(label) for label in state_labels]
            maps=np.stack([np.fft.fftshift(components['magnetization'][state])[roi] for state in order])
            grid(maps,title+': physical magnetization',prefix+'magnetization.png','RdBu_r',-1,1)
        elif components is not None:
            print(f'Mode {mode+1}: residual mode; no physical magnetization map assigned.')
        del objects

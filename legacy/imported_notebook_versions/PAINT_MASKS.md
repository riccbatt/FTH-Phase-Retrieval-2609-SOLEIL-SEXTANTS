# Defining masks with Microsoft Paint

Both detector-pixel masks and real-space support masks can be made by painting
bright-red pixels in a PNG. The PNG must keep its original pixel dimensions.
Use the Pencil tool or another hard-edged tool; do not resize, crop, rotate, or
save as JPEG.

## Detector pixel mask

1. Before FTH, run the export cell in `00_define_mask_pixel_paint.ipynb`.
2. Open this generated reference image in Paint:
   `processed/mask_pixels/mask_pixel_<image-id>_reference.png`.
3. Paint every unusable detector pixel pure bright red (RGB `255, 0, 0`).
4. Use **Save as > PNG** and save the edited image as:
   `processed/mask_pixels/mask_pixel_<image-id>.png`.
5. Run the final import/validation cell in the same notebook.

Notebook 01 also loads this same painted file when the corresponding
`mask_id` is selected in `hologram_inputs`. A value of 1 in the resulting
`mask_pixel` array means that the detector pixel is excluded. Existing legacy
`mask_pixel_<image-id>.npy` files are still accepted if no PNG exists.

If several images are stitched, define a mask for every input first. The
stitched mask is their intersection: a pixel stays masked only when it was
covered in every aligned input image.

## Support mask

Paint is one of three equivalent ways to produce the support mask in notebook
03. You may instead paint a Napari labels layer, or define circular supports as
`(y, x, radius)` coordinates. Run only the section for the method you choose,
then continue with the common ROI and save sections.

1. Run notebook 03 through the support-preview and **Paint support mask with
   PNG** export cells.
2. Open:
   `processed/supportmask/supportmask_<image-id>_reference.png`.
3. Paint the real-space region that belongs to the object pure bright red
   (RGB `255, 0, 0`).
4. Save the edited image, as PNG, under the exact name:
   `processed/supportmask/supportmask_<image-id>.png`.
5. Run the following import and save cells in notebook 03.

A value of 1 in `supportmask` means that the real-space pixel is inside the
allowed reconstruction support.

## Important details

- The image ID is printed by each export cell; replace `<image-id>` with that
  number, without angle brackets.
- Paint in bright red. The loader accepts red channel values 200–255 only when
  both green and blue are 0–80.
- Keep the canvas size exactly unchanged. The loader reports an error if the
  painted PNG and reconstruction data have different shapes.
- The `_reference.png` file is only a visible background. The filename without
  `_reference` is the mask input read by the notebooks.
- Final FTH and phase-retrieval figures remain directly in `processed/`.
  Mask inputs and outputs are separated into `processed/mask_pixels/` and
  `processed/supportmask/`.

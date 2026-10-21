# Data and third-party attribution

## Ultrasound demonstration images

The 34 demonstration images are derived from **UMUD benchmark muscle architecture dataset, version 0.1.0**, in the **UMUD Data Repository**:

- Repository: https://osf.io/xbawc/
- Source archive: https://osf.io/download/hb5xj/
- File metadata: https://api.osf.io/v2/files/677b8a2ebff45e172c698203/
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/
- License record linked by the source OSF node: https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/

Credit: the UMUD dataset contributors and expert raters. Related project publication:

Paul Ritsche, Fabio Sarto, Francesco Santini, Christoph Leitner, Martino V. Franchi, Oliver Faude, Taija Finni, Olivier Seynnes and Neil Cronin. **UMUD: a web application for easy access to musculoskeletal ultrasonography datasets.** BMC Medical Imaging 26, 139 (2026). https://doi.org/10.1186/s12880-026-02170-0

Modifications: TIFF images converted to grayscale PNG for browser display; display resizing; colored model masks, boundary fits, measurements and extrapolation lines added as separate visualization layers. Expert image-level values are summarized for exploratory comparisons. No endorsement by the original authors is implied. Original dataset terms continue to apply to the images and source annotations. This is not a redistribution of the entire Kaggle competition dataset.

## Research and software references

- Ritsche, Seynnes and Cronin. **DL_Track_US: a python package to analyse muscle ultrasonography images.** JOSS 8(85), 5206 (2023). https://doi.org/10.21105/joss.05206
- Segmentation Models PyTorch: https://github.com/qubvel-org/segmentation_models.pytorch
- PyTorch: https://pytorch.org/

Dependencies retain their own licenses. Third-party repositories and pretrained weights are not vendored here. This snapshot does not specify a new open-source license for the project code.

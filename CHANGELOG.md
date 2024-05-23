## [0.1.0] - Unreleased

### Added

- [DFIS-1170] Upgrades to `dfipy==6.0.1` and adds static-analysis and linting.
- [DFIS-541] Added example notebook based on pending MR on dfilib.
- [DFIS-542] First version.
- [DFIS-542] Added ability for users to create polygons from Mapbox.
- [DFIS-587] Added 2 notebooks; 1 showcasing the numerous permutations of DFI queries; 2 showcasing an analysis of similar entities, that make hundreds to thousands of DFI queries.
- [DFIS-598] Now using dfipy from public-pypi instead of using private artifactory.
- [DFIS-611] doc update and library import in each notebook if we are running from colab.
- [DFIS-709] Created GeoLife and London traffic notebooks for public use in ./examples. Moved all other notebooks to ./private examples. Updated Readme file to be suitable for publishing.
- [DFIS-709] Updated the GeoLife and London demo. Added configurations to KeplerGP maps for consistency and readability. Improved text of comments and headings of notebooks.
- [DFIS-709] Added tutorial to draw polygons on a map and use them to query the DFI.
- [DSCI-39] Code for blog post added as part of the documentation of dfipy examples.

### Changed

- [DFIS-1948] Update references to developer site.
- [DFIS-1276] Upgrades to `dfip==9.0.1`.
- [DFIS-810] Link to live documentation added to the README.
- [DFIS-789] Kepler config files are now loaded from json files.

### Fixes

- [DFIS-1559] Fixed broken Github links for MyBinder.
- [DFIS-748] Broken import and removes unused imports.
- [DSCI-53] Blog post typo in module noisyfier.
- [DFIS-1166] Fixed query failing due to Payload now removed from DFI API.

### Removed

- [DFIS-1032] Remove deprecated polygons endpoint.
- [DFIS-1933] Removed Geolife references. Dataset is no longer available.

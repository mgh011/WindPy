
tool_box:
  - nandetrend

qualityCheck:
  - stationary_test
  - saturation_test

windFiltering:
  - despike_mauder

tiltCorrection:
  - static_corr
  - dynamic_corr
  - planar_fit (Work in progress)

windFeatures:
  - friction_velocity
  - obukhovLength

windSpectra:
  - fft 
  - plot_fft 
  - psd
  - plot_psd

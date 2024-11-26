
tool_box:
  - nandetrend

qualityCheck:
  - stationary_test
  - saturation_test

windFiltering:
  - despike_mauder (V)

tiltCorrection:
  - static_corr
  - dynamic_corr
  - planar_fit (Work in progress)

windTurbulence:
  - momentumAndHeatFlux
  - frictionVelocity (V)
  - obukhovLength (V)
  - fluxUncertainty
  - variableUncertainty

windSpectra:
  - fft 
  - plot_fft 
  - psd
  - plot_psd

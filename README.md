# Speedy
(Fast estimation of the background of red giant power spectra.)

The idea behing this code is that estimating the backgrounds of red giant (and in general of any asteroseismic) power spectra takes time and generally involves MCMC (as MLE can be temperamental at the best of times). 

Can we use our knowledge of the models already fitted to high-quality datasets to not only to help provide good initial guesses to lower-quality (or shorter length) datasets, but also to provide good estimates of the background of high-quality stars too?

This may sound circular but there is a point to this. When providing initial estimates of asteroseismic parameters, such as delta nu, generally either the power spectrum of the power spectrum (PSPS) or autocorrelation of the power spectrum (ACF) is used. Both of these methods are best suited to being used with the background subtracted or signal-to-noise spectrum (background divided), otherwise unwanted artefacts are seen and the determination is poor. Normally a box-car smoothed version of the power spectrum is used to approximate the stellar background but this will also contain the oscillations and so reduce the signal-to-noise around the oscillation region. In addition the window is generally around 10uHz which is fine for higher numax stars, but for those with a numax comparable (or lower) than the window size, this creates additional problems and ruins the determination of delta nu.

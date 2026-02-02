Ask any clarifying questions before carrying this out.
I want you to queue the following ablations to run overnight, optimising GPU usage so that it runs as fast as possible. Label the experiments appropriately so that they are easily identifiable (e.g. FNO_geom_attn_seed42_mixed_res)

In this file, report the metrics in the muscle region (rel L2, smoothness etc) on both the train and testing sets for: 
 - UNet
 - FNO baseline (with analytical, layers_6)
 - FNO baseline without analytical solution
 - FNO with light geometry attention
 - TFNO

There is 501 low res samples of data, with 400 downsampled high-res samples. These are split 75/15/15 into train, val and test.

I also want a column for mixed resolution training (carried out on all except UNet). This trains on an additional 50 samples, testing on the remaining high-resolution test set.

Each should be run over 10 seeds for uncertainty quantification. The test sets MUST BE THE SAME across seeds. Do all the separate ones before repeating with different seeds. Include a separate table for each seed, with a final one showing the averages (and mean and standard deviations) for each entry.

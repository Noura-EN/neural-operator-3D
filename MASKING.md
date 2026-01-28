There is a codebase here that trains a neural operator to learn a potential field from a source field.

I am unclear on the recent round of ablations and results. I want performance (MSE and NMSE, gradient loss) reported for the entire region and the 90pct and 99pct region, and the 10% and 1% region. Note that it is impossible to have performance in a masked out region, so these should be left blank i.e. The baseline will have results for them all, 99pct for all except the 1% region, 90pct for only the 90% region. What I want is comparison between:

1) A baseline FNO (using best parameters/set-up found in ABLATIONS.md) that masks out the singularity with 3 voxels but otherwise learns the full signal. 
2) A 90pct and 99pct model that masks out the top 10% and 1% around the source respectively
3) A re-normalised 90pct and 99pct. These mask out as in (2), but afterwards re-normalise the ground truth signal based on the average mean and standard deviation across all samples.
4) 

Also summarise in this file how the codebase works. What is the architecture, the inputs, the normalisation steps, the loss term options etc
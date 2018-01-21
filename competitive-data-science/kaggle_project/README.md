# The solution is password protected to avoid spoiling the competition for others :-)

# How to generate the solution

Work through the FEAT_ notebooks to generate the required features. Note that the cells that generate the .csv.gz are commented out. Uncomment them if you do want the files.

Then work through MODEL_final to train the models. **Do not blindly run the notebooks!**
Throughout the MODEL notebook there are a series of checkpoints that will save the progress so far (To h5 or pickle). This is intended to be used if you don’t have a lot of RAM. You can just work up to that point, restart the notebook, run Cell 1 to import packages, then scroll down, and reload what you just saved, to wipeout unwanted memory.

In the model notebook, you will first have to train the models once on the training set (This generates the ALT_*_TRAIN) files, and then the ALT_MODEL files, which are the final models. These final models are included. The stacked model tuning is done using the TRAIN models, but at the end you will train the meta-model on the full models.

Finally, to predict run the Predict notebook **(again, not blindly).** Optionally you can try to zero out some predictions uncommenting one of the final lines, but my best score was achieved with the results “as they are”.

The final solution should score slightly below 0.95, achieving 10/10 in the grader.

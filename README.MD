For this challenge we are going to run a couple of hypothetical scenarios. Each scenario mirrors the types of responsibilities you would assume for the position within our team.

The position includes:
* A data labelling component. You will be expected to review and assess the usability of the data processed by our labelling team. Your assessments must guide the team to labelling the data with increasing accuracy and decreasing error. These assessments will always include a qualitative and quantitative component, which must be backed up with statistics.
* A data generation component. In addition to the standard image augmentation operations we produce our own in house algorithms to create synthetic data. Your responsibilities would include implementing and testing these algorithms and their outputs.
* A data management component. After all the data labelling and generation is done there is always a tonne of data to store and curate. Data, processed and unprocessed, must be easily accessible, referenceable and backed up on the cloud.

# Data Labelling - Labeller Assessment
To asses the quality of the annotations, I have created annotations of my own. I then compare them with the given annotations using the script 
```python
mask_evaluation.py
```
The metrics I used are: Jaccard(IoU), F1 (Dice), Recall, Precision and Accuracy. These metrics give an indication of the quality of the given mask compared to the desired mask I created. I mainly used the Jaccard score. If for instance the annotation has excessive padding or contains other images that are not of dogs, the Jaccard score decreases. High quality annotations are indicated by a Jaccard score of close to 1.0. The results can be found in the  file: `mask.json`.

# Data Generation - Working with Images
To extract the annotated areas from the raw files, I first turned the mask into binary, then inverted it. I then I multiplied with the raw image. I only used masks that have a Jaccard score of 85% or higher. This was done with the script

```python
extract_doggy_regions.py
```


# Data Management - Combining images

To collate the images I build several functions to resize and concat images horizontally and veritically. I then combined the images to build the collated image called `collage.png`using the 

```python
extract_doggy_regions.py
```
```python
consolidated_doggy_region.py
```
script.

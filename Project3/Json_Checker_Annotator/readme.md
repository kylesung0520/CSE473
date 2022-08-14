This json file checker is to check the format of json file to see if it meets the requirement.
This tool can also annotate your findings on the photos if you provide the images.
In order to run the script, 
First, cd [Json_Checker_Annotator] 
Then, you need to put your [results.json] under the json folder and
then run the [json_checker_annotator.py] script . This script will then print out the results
of the check.

If you also would like to overlay your detected boxes with the corresponding images, you
can put all your tested images under imgs folder, the script will then help draw boxes
over the photos based on your results.json. The annotated images will be saved to the
[annotated] folder.
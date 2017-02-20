# SED Evaluation Pipline



## sed_evaluation.py

* Pipline file
* prepare_csv(): Generate csv file to ./csv/*.csv from result_{class}.txt
* xml_script(): Generate xml file to ./xml/*.xml from ./csv/\*.csv
* exp_control(): Run experiment defined in queue[i] (define test videos), save results to result/dirnames[i]/
* If you have csv file or xml file, comment corresponding functions



## Parameters

* validator_path: F4DE SED08 TV08ViperValidator tool
* cams: all videos
* CLASSES: classes to evaluate. If you need to evaluation other classes than Pointing, CellToEar, Embrace, regenerate ground truth file use prepare_gtf() & gtf_script() functions
* dettemplate: results file template. 
* threshold: threshold for detection score, used in post process.


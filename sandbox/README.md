Let's make fhis folder as a sandbox.

Use segments of videos and tests

Build the logic on the same data


# some ideas not to forget to explore

- adopt a flexible cropping region (it's not always bottom): came acroos [EAST text detector method](https://arxiv.org/abs/1704.03155v2)

- more suitable/optiimized SAMPLE_RATE (instead of just X frames per second): automatically detect frame change in the cropped area intead of naïvely OCR’ing :) the whole frame every time.

- De-duplicate: after OCR find a better way to compare the text string to the previous one. options i think of include:
    - exact match 
    - simple [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
    - SequenceMatcher threshold (the one i tested)

- automatically calibrate OCR-extracted text using LLL (ex: Gemini, gpt):  somtimes there are interferences such as scene and screen clarity, etc. use llm to further improve the recognition ability of OCR in subtitle extraction (proofread, recognize wrong words or sentences and correct them according to the context to improve the accuracy of the text).

- find a way to get rid of watermarks if prresent (clideo.com for ex.)

- ah yes another thing: at which step should we crop?
    - extract then directly **crop frames** before filtering and do ocr
    - or extract frames, filter duplicates then **crop frames** and do ocr

- maybe explore multiprocessing/multithreading???

- backbones / tools ideas:
    - [doctr](https://github.com/mindee/doctr)

- Florence is not that slow (as i thought): we can envisage finetuning it to better OCR french writings
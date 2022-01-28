import pandas as pd
import numpy as np
import PyPDF2
import tika
import textract
import pdftotext
import os, re

# TODO:
    # further explore possibility of extracting DMS coords with py
    # (currently, textract and pdftotext seem most promising, but
    # both render the degree symbol as `8` for the Gliricidium example PDF
    # and as `\x03` for the Organic... example PDF

# params
n_dec_pts_coords = 2

# regex patterns
decdeg_patt = ''

dms_subpatt = r'\d{1,3}°\s?\d{1,2}\.?\d*[\'’´]\s?\d*\.?\d*[\'’´\"]*'
dms_patt = re.compile(r'%s,?\s?[NSEW]?\s?%s,?\s?[NSEW]?' % (dms_subpatt,
                                                            dms_subpatt),
                      flags=re.UNICODE)


# data structures
lats = []
lons = []

# get list of PDFs
aut_dir = './automatically_downloaded_papers'
man_dir = './manually_downloaded_papers'
files = [os.path.join(aut_dir, f) for f in os.listdir(aut_dir)]
files.extend([os.path.join(man_dir, f) for f in os.listdir(man_dir)])
pdfs = [f for f in files if os.path.splitext(f)[1].lower() == '.pdf']

# parse lats and lons from each PDF
for pdf in pdfs:

    # create a pdf file object 
    pdf_file_obj = open(pdf, 'rb')

    # create a pdf reader object 
    reader = PyPDF2.PdfFileReader(pdf_file_obj)

    # get number of pages in pdf file 
    n_pages = reader.numPages

    # parse coords out of each page
    for pg in range(n_pages):

        # create a page object 
        page_obj = reader.getPage(pg)

        # extract text from page 
        txt = page_obj.extractText()

        print(txt)

    # close the pdf file object 
    pdf_file_obj.close()

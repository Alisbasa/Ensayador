
!pip install datasets
!pip install apache_beam mwparserfromhell

from datasets import load_dataset

dataset=load_dataset("wikipedia", language="es", date="20221101", beam_runner="DirectRunner")

dataset.to_pandas("./")
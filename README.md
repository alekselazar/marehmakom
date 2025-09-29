# MarehMakom
Finding quotes in Torah and Rabbinical texts is challanging task. For this task I propose LaBSE+FiLM+U-Net span finder for Torah/Talmud quotations. My model encodes query and target texts with LaBSE model, and then uses FiLM+UNet for segmentation task, returning quote mask, where 1 is probably a quote token

## Data
As dataset I used two sources: real quotes in Rashi commentaries from Tanakh and syntetic data composed from Talmud and Tosafot texts. In `/data` dir in this repository you can see a code, used to construct a dataset.

## Training
This model was trained for 15 epochs, in the end here are metrics of training and validation:
### Last Epoch training metrics:
|F1|P|R|LOSS
|--|--|--|--|
| 0.9728|0.9577|0.9884|0.0033
### Validation metrics:
I calculated metrics for span(how detected start and end of the quote are accurate) and token level
|span-f1|span-p|span-r|token-f1|token-p|token-r|
|--|--|--|--|--|--|
|0.7720|0.7829|0.7614|0.9172|0.9394|0.8987|

Full training code you can see in `/train` dir in this repository

# Install

```bash
pip install git+https://github.com/youruser/marehmakom.git
```

After initial install it will download files for tokenizer and model. If you want to fine-tune model and then use new files or just experiment, you can replace original files in cache dir with your files.
 
# Use

## As library:
```python
from marehmakom import load

infer = load()
out = infer.predict("אמר רבי עקיבא", "…long daf content…")
print(out["snippets"])
```
## CLI:
```
marehmakom-infer --query "אמר רבי עקיבא" --target "$(cat page.txt)" --json
```
# Credit
This work is inspired by [Sefaria](https://www.sefaria.org.il/team) incredible team during working on my internship project. Thank you, great people working on great purpose.
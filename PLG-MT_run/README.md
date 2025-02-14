# Couple Learning(PLG-MT)



## Steps of training  
- open the `config.py` file to confirm parameter of mode is `""`.
- execute `python MT_train.py`, Complete the training of the baseline model.
- execute `python PLG_process.py`, Complete the generation of pseudo labels
- open the `config.py` file to confirm parameter of mode is `"_TbS"`.
- execute `python MT_train.py -sm "_TbS"`, Complete the training of the PLG-MT model.


## Testing 
### mean teacher only
```bash
python TestModel.py -m "stored_data/MeanTeacher/model/baseline_best" -g ../dataset/public_eval/metadata/eval/public.tsv  \
-ga ../dataset/public_eval/audio/eval/public -s stored_data/PLG_MT_public_test/public_predictions.tsv
```

### PLG-MT model
```bash
python TestModel.py -m "stored_data/MeanTeacher_TbS/model/baseline_best" -g ../dataset/public_eval/metadata/eval/public.tsv  \
-ga ../dataset/public_eval/audio/eval/public -s stored_data/mean_teacher_public_test/public_predictions.tsv
```

### Results in paper
baseline model path : `stored_data/models/trained_baseline_model`

PLG-MT model path : `stored_data/models/trained_PLG-MT_model`

To reproduce the results of the paper run

```bash
python TestModel.py -m "stored_data/models/trained_PLG-MT_model" -g ../dataset/public_eval/metadata/eval/public.tsv  \
-ga ../dataset/public_eval/audio/eval/public -s stored_data/mean_teacher_public_test/public_predictions.tsv
```

## Test result

###  public set
|                       | EB-F1     | PSDS_1    |PSDS_2     |PSDS_3     |
|-----------------------|----------:|----------:|----------:|----------:|
| baseline              | 37.12%    | 0.5900    | 0.5070    | 0.4209    |
| PLG-MT                | 39.41%    | 0.6176    | 0.5552    | 0.4609    |

###  validation set
|                       | EB-F1     | PSDS_1    | PSDS_2    | PSDS_3    |
|-----------------------|----------:|----------:|----------:|----------:|
| baseline              | 32.39%    | 0.5831    | 0.4916    | 0.4098    |
| PLG-MT                | 33.37%    | 0.6116    | 0.5420    | 0.4398    |


## Supplementary experimental results

### EB-F1(%) results with different methods on different test set

|                       | Validation     | public eval  |
|-----------------------|----------:|----------:|
| CRNN              |    28.14 |   33.56  | 
| + Mean Teacher(Baseline)                 |  32.39   |  37.12   | 
| + PLG                 |  30.04   |  35.26   |
| + Mean Teacher + PLG                 |   33.93 |  39.18   |


###  EB-F1(%) with different pseudo-labels and PLG models on validation set
|                       | FBCRNN     | MT-CRNN  |
|-----------------------|----------:|----------:|
| Baseline              |    32.39 |   32.39  | 
| + UPW                 |  33.95   |  30.06   | 
| + WPS                 |  35.80   |  32.15   |
| + UPW                 |   37.24   |  32.42   |
| + UPS + WPS                 |  37.97  | 33.52   |
| + UPS + WPS + UPW                 | 41.33   |  33.93   |


### References
 - [[1]] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.
 - [[2]]  Tarvainen, A. and Valpola, H., 2017.
 Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results.
 In Advances in neural information processing systems (pp. 1195-1204).

[1]: http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Delphin_15.pdf
[2]: https://arxiv.org/pdf/1703.01780.pdf
[4_separate_mixtures.sh]: ../scripts/4_separate_mixtures.sh

[dcase2019-baseline]: https://github.com/turpaultn/DCASE2019_task4
[model-weights]: https://doi.org/10.5281/zenodo.3726375

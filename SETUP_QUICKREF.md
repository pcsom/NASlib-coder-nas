# NASLib Setup

Full setup instructions: https://docs.google.com/document/d/1FEsUgWnE2B6WFuZfNILubmfryQmkrSWv3xWnHm0PdiA/edit

## Activate Environment
```bash
module load anaconda3
conda activate naslib39v2
```

## Verify
```bash
./verify_setup.sh
```

## Download Benchmark Data

Place all files in `naslib/data/`

**NB201:**
- https://drive.google.com/file/d/1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ/view
- https://drive.google.com/file/d/1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp/view
- https://drive.google.com/file/d/1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH/view

**NB301:**
- Data: https://drive.google.com/file/d/1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF/view
- Models: https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061510
  - After extracting: `mv naslib/data/nb_models naslib/data/nb_models_1.0`

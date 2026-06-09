
# MNIST Digit Classifier

A web app for handwritten digit recognition, built with a CNN trained on MNIST and served via Streamlit.

## Features

- Draw a digit on a canvas and get an instant prediction
- CNN model trained with PyTorch (98%+ accuracy on MNIST test set)

## Files

| File | Description |
|------|-------------|
| `mnist_app.py` | Streamlit app (canvas input + CNN inference) |
| `mnist_train.py` | Training script |
| `mnist_cnn.pt` | Trained model weights |
| `mnist_training_report.txt` | Training metrics |

## Run locally

```bash
pip install -r requirements.txt
streamlit run mnist_app.py
```

## Deploy on Streamlit Cloud

Push this repo to GitHub, then connect it on [streamlit.io/cloud](https://streamlit.io/cloud).  
Entry point: `mnist_app.py`

## Requirements

Python 3.12 — see `requirements.txt` (PyTorch CPU build).


url: https://ml01webappmnsit-hxmo9mtkxgbpqd5weuawk6.streamlit.app/

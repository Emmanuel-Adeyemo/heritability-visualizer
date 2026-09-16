# Heritability — Teaching App
 
Streamlit app for teaching heritability in a graduate plant breeding course at Auburn University in Fall 2026.

### 2. Breeder's Equation Explorer (`app.py`)
Teaches R = h² × S and how response to selection behaves across generations.
 
- **Panel A** — one generation of truncation selection: parent population, selected tail, and shift in the offspring pop mean.
- **Panel B** — response projected across multiple generations.
- **Panel C** — realized heritability.
Core simulation logic lives in `simulate.py`.


### Running locally
 
```bash
pip install -r requirements.txt
streamlit run app.py             # Parent-Offspring Regression
```

### App deployed on Streamlit: [aub.ie/heritability](https://aub.ie/heritability)
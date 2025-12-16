~250mb Of Space Needed For Full Functionality 

(If Dependencies Are Not Already Installed)

Requirements:


```
pkg update && pkg upgrade
# General build tools and Python development headers
pkg install clang make python libjpeg-turbo libpng zlib openssl git
```
```
pip install instaloader Pillow phonenumbers pycountry networkx
pip install scikit-learn numpy scipy # Required for sklearn and its dependencies
pip install spacy textblob pytz timezonefinder reportlab

# Download the required spaCy model
python -m spacy download en_core_web_sm

# Download the required nltk data
python -c "import nltk; nltk.download('stopwords')"
```


Installation:


```

pkg install git python -y && \
pip install requests stdiomask && \
git clone https://github.com/Yourl0calbroker/InstaOSNIT.git $HOME/InstaOSNIT && \
chmod +x $HOME/InstaOSNIT/APITrace.py $HOME/InstaOSNIT/InstaOSNIT.py && \
mkdir -p $HOME/bin && \
if ! grep -q 'export PATH="$HOME/bin:$PATH"' $HOME/.bashrc; then echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc; fi && \
source $HOME/.bashrc && \
ln -sf $HOME/InstaOSNIT/APITrace.py $HOME/bin/APITrace.py && \
ln -sf $HOME/InstaOSNIT/InstaOSNIT.py $HOME/bin/InstaOSNIT.py

```


Run The Following Commands To Use:

```
APITrace.py
```
```
InstaOSNIT.py
```

If That Does Not Work Use:

```
cd InstaOSNIT
```
```
python3 ./InstaOSNIT.py
```

or

```
python3 ./APITrace.py
```
Advanced Runs:

```
# Basic run with Instaloader login (recommended for best results)
python3 InstaOSNIT.py -t <TARGET_USERNAME> --login-user <YOUR_IG_USER> --login-pass <YOUR_IG_PASS> --output <TARGET_USERNAME>_deep.json --format json
```
```
# Run with an existing sessionid (for API access without Instaloader login)
python3 InstaOSNIT.py -t <TARGET_USERNAME> -s <YOUR_SESSION_ID>
```
```
# Run with deep network analysis and temporal clustering, exporting to GEXF
python3 InstaOSNIT.py -t <TARGET_USERNAME> --login-user <USER> --login-pass <PASS> --deep-network --cluster-temporal -f gexf
```

Updating:


```

cd InstaOSNIT && git stash && git pull && git stash pop

```

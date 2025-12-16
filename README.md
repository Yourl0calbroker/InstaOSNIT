~250mb Of Space Needed For Full Functionality 

(If Dependencies Are Not Already Installed)

Requirements:
```
# 1. Update Termux base system and install essential tools
pkg update && pkg upgrade -y && pkg install -y python python-pip git build-essential libxml2 libxslt libjpeg-turbo nano

# 2. Install all required and optional Python dependencies and upgrade them if they exist
pip install --upgrade requests instaloader Pillow textblob phonenumbers pycountry networkx spacy scikit-learn pytz reportlab nltk
```
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
```
pkg update && pkg upgrade -y
pkg install -y python python-pip git build-essential libxml2 libxslt libjpeg-turbo nano # Base system tools
pkg install -y freetype zlib clang # Essential build libraries for Pillow and other dependencies

```
```
# This line installs all Python dependencies, including the failed ones
pip install --upgrade --no-cache-dir requests instaloader Pillow textblob phonenumbers pycountry networkx spacy scikit-learn pytz reportlab nltk

```

```
# 1. Download spaCy Model (if you get "Module named spacy" error, this is the fix)
python -m spacy download en_core_web_sm

# 2. Download NLTK data
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


# 1. Basic, Unauthenticated Scan (Public Data Only)
```
python InstaOSNIT.py --target <target_username> --post-limit 50 --format json
```
# 2. Authenticated Scan (Recommended for Full Data Access)
```
python InstaOSNIT.py --target <target_username> --login-user <your_username> --login-pass <your_password> --post-limit 300 --format json
```
# 3. Session ID Scan (API Focus)
```
python InstaOSNIT.py --target <target_username> --sessionid <YOUR_SESSION_ID_COOKIE> --post-limit 150 --format json
```
# 4. Deep Network & Temporal Clustering (Best Value - Requires Login)
```
python InstaOSNIT.py --target <target_username> --login-user <your_username> --login-pass <your_password> --post-limit 500 --deep-network --cluster-temporal --output full_report.json
```
# 5. Targeted Keyword Search (Requires Login)
```
python InstaOSNIT.py --target <target_username> --login-user <your_username> --login-pass <your_password> --terms "London" "finance|investing" "phone number" --format json
```
# 6. GEXF Network Export (for Gephi - Requires Login)
```
python InstaOSNIT.py --target <target_username> --login-user <your_username> --login-pass <your_password> --post-limit 100 --format gexf
```
# 7. PDF Summary Report (Requires Login)
```
python InstaOSNIT.py --target <target_username> --login-user <your_username> --login-pass <your_password> --post-limit 300 --format pdf
```

~250mb Of Space Needed For Full Functionality 

(If Dependencies Are Not Already Installed)

Requirements:


```
pip install instaloader Pillow textblob phonenumbers pycountry networkx spacy timezonefinder scikit-learn pytz reportlab
python -m spacy download en_core_web_sm

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


Updating:


```

cd InstaOSNIT && git stash && git pull && git stash pop

```

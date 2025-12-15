Installation:

```

pkg install git python -y && \
pip install requests stdiomask && \
git clone https://github.com/Yourl0calbroker/InstaOSNITpkg.git $HOME/InstaOSNITpkg && \
chmod +x $HOME/InstaOSNITpkg/APITrace.py $HOME/InstaOSNITpkg/InstaOSNIT.py && \
mkdir -p $HOME/bin && \
if ! grep -q 'export PATH="$HOME/bin:$PATH"' $HOME/.bashrc; then echo 'export PATH="$HOME/bin:$PATH"' >> $HOME/.bashrc; fi && \
source $HOME/.bashrc && \
ln -sf $HOME/InstaOSNITpkg/APITrace.py $HOME/bin/APITrace.py && \
ln -sf $HOME/InstaOSNITpkg/InstaOSNIT.py $HOME/bin/InstaOSNIT.py

```

Run The Following Commands To Use:

```
APITrace.py
```
```
InstaOSNIT.py
```

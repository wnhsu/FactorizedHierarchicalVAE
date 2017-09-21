all: kaldi kaldi_python
.PHONY: kaldi_python kaldi_d1e1e3b kaldi kaldi_github

kaldi: kaldi_github
	cp -r kaldi_github kaldi
	cd kaldi/tools; ./extras/check_dependencies.sh; $(MAKE) all -j 4
	cd kaldi/src; ./configure --shared; $(MAKE) depend -j 8; $(MAKE) all -j 4

kaldi_d1e1e3b: kaldi_github
	cp -r kaldi_github kaldi_d1e1e3b
	cd kaldi_d1e1e3b; git checkout d1e1e3db3400986beadd46b4ca5c7d63ea1af9b4
	cd kaldi_d1e1e3b/tools; ./extras/check_dependencies.sh; $(MAKE) all -j 4
	cd kaldi_d1e1e3b/src; ./configure --shared; $(MAKE) depend -j 8; $(MAKE) all -j 4
	
kaldi_github:
	git clone https://github.com/kaldi-asr/kaldi.git kaldi_github
	
kaldi_python: kaldi_d1e1e3b
	git clone https://github.com/janchorowski/kaldi-python.git kaldi_python
	cd kaldi_python; $(MAKE) all KALDI_ROOT=$(CURDIR)/kaldi_d1e1e3b

clean:
	rm -rf kaldi_d1e1e3b kaldi_python kaldi_github kaldi

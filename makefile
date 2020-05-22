corpus:
	wget -P ./data/ "http://www.nilc.icmc.usp.br/nilc/tools/fapesp-corpora.tar.gz"
	tar -xvzf  ./data/fapesp-corpora.tar.gz -C data
	tar -xvzf ./data/fapesp-corpora/corpora/pt.tgz -C data/fapesp-corpora/corpora/
	python prepare_corpus.py
embeddings:
	wget -P ./data/ "http://143.107.183.175:22980/download.php?file=embeddings/glove/glove_s50.zip" -O glove_s50.zip
	unzip ./data/glove_s50.zip -d ./data
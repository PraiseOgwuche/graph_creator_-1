run:
	streamlit run Graph_Creator.py

reqs:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm

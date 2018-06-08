.PHONY: init install uninstall test

all: install

install:
	#pip install -r requirements.txt --user
	pip install . --user --upgrade

uninstall:
	pip uninstall ae

test:
	nosetests --rednose tests


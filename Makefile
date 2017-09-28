.PHONY: init install uninstall test

all: install

install:
	pip install . --user --upgrade

uninstall:
	pip uninstall ae

test:
	nosetests --rednose tests


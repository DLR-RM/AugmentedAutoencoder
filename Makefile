
all: install

install:
	pip install . --user --upgrade

uninstall:
	pip uninstall ae

test:
	nosetests --rednose tests

.PHONY: init install uninstall test
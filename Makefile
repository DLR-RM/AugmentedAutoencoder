.PHONY: init install uninstall test

all: install

install:
	#pip install -r requirements.txt --user
	pip install . --user --upgrade

uninstall:
	pip uninstall auto_pose

test:
	nosetests --rednose tests


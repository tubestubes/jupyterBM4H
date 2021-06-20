book: 
	runjb book
	cp -rf .\book\_build\html docs
	
gh:
	git add .
	git commit -m "makefile commit"
	git push origin master

source_notebooks = ./examples
source_blog = ./docs/blog

.PHONY: format lint requirements

format:
	poetry run nbqa black $(source_notebooks)
	poetry run nbqa black $(source_blog)
	poetry run nbqa isort $(source_blog)
	poetry run nbqa isort $(source_notebooks)

lint: format
	poetry run nbqa flake8 $(source_notebooks)
	poetry run nbqa flake8 $(source_blog) --exit-zero
	poetry run nbqa mypy $(source_notebooks)
	
	# disabling mypy for blog
	# poetry run nbqa mypy $(source_blog)
	poetry run nbqa pylint $(source_notebooks)
	poetry run nbqa pylint $(source_blog)


# create also the requirement file for binder to build the python environment.
requirements:
	poetry update
	poetry install
	poetry lock
	poetry export -f requirements.txt --output requirements.txt --without-hashes --without-urls

dev: 
	docker build -t dfipy-examples .

	docker run -it --rm \
	-p 8888:8888 \
	dfipy-examples \
	jupyter lab \
		--NotebookApp.default_url=/lab/ \
		--ip=0.0.0.0 \
		--port=8888

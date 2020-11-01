MKDIR_P = mkdir -p
OUT_DIR = DataManager DataManager/PreprocessingStrategy Classifiers Statistician
current_dir = $(shell pwd)

all: ${OUT_DIR}

${OUT_DIR}:
	@${MKDIR_P} $@
	@cd ${current_dir}/../../src/$@; pyreverse -o png -A ./*.py; mv classes.png ${current_dir}/$@/; rm packages.png;
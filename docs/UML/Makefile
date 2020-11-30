MKDIR_P = mkdir -p
OUT_DIR = DataManagement DataManagement/Preprocessing Classifiers "."
current_dir = $(shell pwd)

all: ${OUT_DIR}

${OUT_DIR}:
	@${MKDIR_P} $@
	@cd ${current_dir}/../../; pyreverse -o png -A src/$@/; mv classes.png ${current_dir}/$@/; rm packages.png;
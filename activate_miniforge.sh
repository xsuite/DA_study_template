# Get if docker miniconda exists
FILE=/usr/local/DA_study/miniforge_docker/bin/activate

# Check if  the job is launched from docker
if [[ -f "$FILE" ]];

# source the make_miniforge.sh script, concatenating the path to the script directory
then
    source /usr/local/DA_study/miniforge_docker/bin/activate
# Only activate environment 
else
    source $SCRIPT_DIR/miniforge/bin/activate
fi



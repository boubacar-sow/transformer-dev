#!/bin/bash

RUNNER_PATH="~/.c9/runners/"
SERVICE_PATH=$(pwd)
SERVICE_SLUG=transformer


# Get Workspace Hostname and README file to use
W_README=""
W_LIST=$(cd ~ && ls README.*.md | sed -nr 's/README.(.*).md/\1/p')

echo -e "\e[34mSélectionner le numéro du workspace pour lequel ce projet doit être configuré: \e[0m"
select CHOICE in $W_LIST
do
    if [[ ${W_LIST} =~ (^|[[:space:]])${CHOICE}($|[[:space:]]) ]]
    then
        W_HOSTNAME=${CHOICE}
        W_README=README.${CHOICE}.md
        break
    else
        echo Not an option
    fi
done

[ -z "${W_README}" ] && echo "\e[34mAucun Workspace README trouvé.\e[0m" && exit 1
echo ${W_HOSTNAME} configuration selected

AVAILABLE_PORT=$(cat ~/$W_README | sed -nr "s/^- ([0-9]{4}) .*$/\1/p"  | head -1)

# Write small config for django
cat << EOF > ~/.config/${SERVICE_SLUG}.conf
SERVICE_PORT $AVAILABLE_PORT
WORKSPACE_NAME $W_HOSTNAME
EOF

# Create DB
createdb -U postgres ${USER//_/.}.${SERVICE_SLUG}

if [ -z "$PYPI_PASSWORD" ]
then
    echo "La variable 'PYPI_PASSWORD' n'est pas définie. Executer la commande suivante avec le bon mot de passe:"
    echo " $ export PYPI_PASSWORD=<password>"
    exit 1
fi

# Poetry config
poetry config http-basic.osrdata-pypi gitlab+deploy-token-591582 $PYPI_PASSWORD
poetry install
poetry run ./manage.py check --settings=config.workspace
poetry run ./manage.py migrate --settings=config.workspace

D_PORT=$(shuf -i 10000-65000 -n 1)

## Create runner
mkdir -p ${RUNNER_PATH}

OVERWRITE_RUNNER_CONFIG="x"
if [ -f "${RUNNER_PATH}${SERVICE_SLUG}.run" ]
then
    echo "Une configuration Cloud9 existe déja, écraser ? (y/n)"
    read -n 1 OVERWRITE_RUNNER_CONFIG
    echo

    # User making poor choices
    if [[ ${OVERWRITE_RUNNER_CONFIG} != "y" && ${OVERWRITE_RUNNER_CONFIG} != "n" ]]
    then
        echo "Aucune option ne correspond à ce choix, aucun remplacement effectué"
        exit 1
    fi

    # User chooses to not overwrite
    if [[ ${OVERWRITE_RUNNER_CONFIG} == "n" ]]
    then
        exit 0
    fi
fi

# Get PoetryEnv path
PYENV_PATH=$(cd ${SERVICE_PATH} && echo $(poetry show -v) | head -1 | awk '{print $3}')

cat << EOF > ~/.c9/runners/${SERVICE_SLUG}.run
// This file overrides the built-in Python 3 runner
// For more information see http://docs.aws.amazon.com/console/cloud9/change-runner
{
  "script": [
    "if [ \"\$debug\" == true ]; then ",
    "    ${PYENV_PATH}/bin/python -m ikp3db -ik_p=${D_PORT} -ik_cwd=\$project_path \"\$file\" \$args",
    "else",
    "    ${PYENV_PATH}/bin/python \"\$file\" \$args",
    "fi",
    "checkExitCode() {",
    "    if [ \$1 ] && [ \"\$debug\" == true ]; then ",
    "        python3 -m ikp3db 2>&1 | grep -q 'No module' && echo '",
    "    To use python debugger install ikpdb by running: ",
    "        sudo yum update;",
    "        sudo yum install python36-devel;",
    "        sudo pip-3.6 install ikp3db;",
    "        '",
    "    fi",
    "   return \$1",
    "}",
    "checkExitCode \$?"
  ],
  "python_version": "python3",
  "working_dir": "\$project_path",
  "debugport": ${D_PORT},
  "\$debugDefaultState": false,
  "debugger": "ikpdb",
  "env": {
    "PYTHONPATH": "${PYENV_PATH}/bin/python"
  },
  "trackId": "${SERVICE_SLUG}"
}
EOF


read -r -d '' RUN_CONFIG << EOF
{
    "args": [
        "runserver"
    ],
    "command": "${SERVICE_PATH}/manage.py runserver ${AVAILABLE_PORT}",
    "debug": false,
    "django": true,
    "env": {
        "DJANGO_SETTINGS_MODULE": "config.workspace",
        "VIRTUAL_ENV": "${PYENV_PATH}"
    },
    "name": "${SERVICE_SLUG}",
    "request": "launch",
    "runner": "${SERVICE_SLUG}",
    "type": "python3"
}
EOF

PROJECT_SETTINGS=$(jq ".run.configs.\"json()\".${SERVICE_SLUG} = ${RUN_CONFIG}" ~/.c9/project.settings)
echo ${PROJECT_SETTINGS} > ~/.c9/project.settings

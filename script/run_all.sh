while getopts d:c:s:p:o: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "s" ) FLG_S="TRUE" ; VALUE_S="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
    "o" ) FLG_O="TRUE" ; VALUE_O="$OPTARG" ;;
  esac
done

cp -f 11e625050a0850c9066c378d9e05f08a/run.sh run.sh
chmod 777 run.sh

./run.sh -t FedMD -d ${VALUE_D} -a ptbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
./run.sh -t FedMD -d ${VALUE_D} -a tbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
./run.sh -t FedGEMS -d ${VALUE_D} -a ptbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
./run.sh -t FedGEMS -d ${VALUE_D} -a tbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
./run.sh -t DSFL -d ${VALUE_D} -a ptbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
./run.sh -t DSFL -d ${VALUE_D} -a tbi -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b 0
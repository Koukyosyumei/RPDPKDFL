while getopts t:d:a:c:f:s:p:o:b: OPT
do
  case $OPT in
    "t" ) FLG_T="TRUE" ; VALUE_T="$OPTARG" ;;
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "a" ) FLG_A="TRUE" ; VALUE_A="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
    "f" ) FLG_F="TRUE" ; VALUE_F="$OPTARG" ;;
    "s" ) FLG_S="TRUE" ; VALUE_S="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
    "o" ) FLG_O="TRUE" ; VALUE_O="$OPTARG" ;;
    "b" ) FLG_B="TRUE" ; VALUE_B="$OPTARG" ;;
  esac
done

echo "fedkd_type is $VALUE_T"
echo "dataset is $VALUE_D"
echo "attack_type is $VALUE_A"
echo "client_num is $VALUE_C"
echo "use_finetune: $VALUE_F"
echo "tempreature: $VALUE_S"
echo "setting: $VALUE_B"

cp -f 11e625050a0850c9066c378d9e05f08a/main.py main.py
python main.py -t ${VALUE_T} -d ${VALUE_D} -a ${VALUE_A} -c ${VALUE_C} -f ${VALUE_F} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b ${VALUE_B}
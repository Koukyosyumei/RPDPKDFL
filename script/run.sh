while getopts t:d:a:c:s:p:o:b: OPT
do
  case $OPT in
    "t" ) FLG_T="TRUE" ; VALUE_T="$OPTARG" ;;
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "a" ) FLG_A="TRUE" ; VALUE_A="$OPTARG" ;;
    "c" ) FLG_C="TRUE" ; VALUE_C="$OPTARG" ;;
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
echo "tempreature: $VALUE_S"
echo "setting: $VALUE_B"

ls
python script/main.py -t ${VALUE_T} -d ${VALUE_D} -a ${VALUE_A} -c ${VALUE_C} -s ${VALUE_S} -p ${VALUE_P} -o ${VALUE_O} -b ${VALUE_B}
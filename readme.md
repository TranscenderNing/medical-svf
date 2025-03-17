nohup python svf_medical.py --batch_size 8 --custom_prefix "骨科康复" > medical_guke.log 2>&1 &
nohup python svf_medical.py --batch_size 8 --custom_prefix "脊髓损伤康复" > medical_jisui.log 2>&1 &
nohup python svf_medical.py --batch_size 2  --custom_prefix "内科康复" > medical_neike.log 2>&1 &
nohup python svf_medical.py --batch_size 2  --custom_prefix "言语、吞咽康复" > medical_yanyu.log 2>&1 &
nohup python svf_medical.py --batch_size 8  --custom_prefix "卒中康复" > medical_zuzhong.log 2>&1 &

nohup python svf_medical.py --batch_size 16  --custom_prefix "dispatch" > medical_dispatch.log 2>&1 &
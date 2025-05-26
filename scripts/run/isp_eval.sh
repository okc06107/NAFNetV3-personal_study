#python isp/run_sidd_benchmark.py -opt /home/swhong/01_NAFNet/AIISP2024/options/test/NAFNetV3/shuffle2-shuffle1-width16.yml
#python isp/run_sidd_benchmark.py -opt /home/swhong/01_NAFNet/AIISP2024/options/test/SIDD/NAFNetv2-width64-mid0p16.yml
#python isp/run_sidd_benchmark.py -opt /home/swhong/01_NAFNet/AIISP2024/options/test/SIDD/NAFNetv2-width90-mid0p16.yml
#python isp/run_sidd_benchmark.py -opt /home/swhong/01_NAFNet/AIISP2024/options/test/SIDD/NAFNetv2-width128-mid0p16.yml
#python isp/run_sidd_benchmark_param.py --template options/test/SIDD/NAFNetv2_template.j2 --ch 32 --mid 9 --sc0 1
#python isp/run_sidd_benchmark_param.py --template options/test/SIDD/NAFNetv2_template.j2 --ch 32 --mid 9 --sc0 2
#python isp/run_sidd_benchmark_param.py --template options/test/SIDD/NAFNetv2_template.j2 --ch 32 --mid 9 --sc0 1



python isp/run_sidd_benchmark_param.py --template options/test/NAFNetV3/template_final.j2 --

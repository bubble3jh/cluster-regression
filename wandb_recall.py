import wandb 

wandb.login()
# 프로젝트와 엔터티 이름 설정
project_name = "cluster-regression"
entity_name = "mlai_medical_ai"

api = wandb.Api()
run = api.run(f"{project_name}/uqjjp45e")

# WandB 로그 데이터 불러오기
history = run.history()

# WandB 로그에서 특정 키에 대한 값 가져오기 (예: 'param_stats/...')
param_keys = [key for key in history.columns if key.startswith("param_stats/")]

# 파라미터 통계 키를 사용하여 모델 구조 추측
model_structure = {}
layer_names = set()

for key in history.keys():
    if key.startswith("param_stats/"):
        # 'param_stats/{layer_name}_...' 형식에서 레이어 이름 추출
        parts = key.split('/')[1].split('_')
        layer_name = '_'.join(parts[:-1])  # 마지막 부분(max, mean, min, variance) 제외
        layer_names.add(layer_name)

# 추출된 레이어 이름 출력
for name in layer_names:
    print(name)
# for key in param_keys:
#     # 'param_stats/{layer_name}_{param_stat}' 형식에서 레이어 이름 추출
#     layer_name = key.split('/')[1].split('_')[0]
#     if layer_name not in model_structure:
#         model_structure[layer_name] = []
#     # 파라미터 통계 종류 추출 (min, max, mean, variance)
#     param_stat = key.split('_')[-1]
#     model_structure[layer_name].append(param_stat)

# # 모델 구조 출력
# for layer, stats in model_structure.items():
#     print(f"Layer: {layer}, Stats: {stats}")

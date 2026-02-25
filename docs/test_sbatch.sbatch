#!/bin/bash

#SBATCH -N 2                       # 노드 수: 2개 노드 사용
#SBATCH --job-name=dli_firstSlurmJob  # 작업 이름
#SBATCH -o /dli/nemo/logs/%j.out   # 표준 출력 로그 파일 (%j = job ID)
#SBATCH -e /dli/nemo/logs/%j.err   # 표준 에러 로그 파일

srun -l /dli/code/test.sh          # 실제 실행할 명령어

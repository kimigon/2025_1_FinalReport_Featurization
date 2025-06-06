
from math import floor
import numpy as np
import math
import ast
import pandas as pd
import statistics
from scipy.stats.mstats import gmean
import time
import matplotlib.pyplot as plt
from ase.io import read, write
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifParser
from pymatgen.core.structure import Structure
from matminer.featurizers.site import CoordinationNumber
from matminer.featurizers.structure import StructuralComplexity
from matminer.featurizers.composition.element import BandCenter
from pymatgen.core import Composition

cn_featurizer = CoordinationNumber.from_preset('VoronoiNN')
band_center_featurizer = BandCenter()
sc_featurizer = StructuralComplexity(symprec=0.1)

with open('atomic_features/firstionizationenergy.txt') as f:
    content = f.readlines()
fie = [float(x.strip()) for x in content]

with open('atomic_features/mendeleev.txt') as f:
    content = f.readlines()
mendeleev = [float(x.strip()) for x in content]

def featurization(slab, bottom=False, tol=0.7):
    """
    메모리 사용량을 최소화하여 슬래브 구조를 featurize하는 함수.
    - slab: 입력 구조 (파일 경로, pymatgen Structure, ASE Atoms, dict 등 허용)
    - bottom: True면 아래쪽 표면, False면 위쪽 표면을 계산
    - tol: 층 구분 시 z 좌표 차이 허용 오차
    반환값: [error_msg, f_chi, f_chi2, f_chi3, f_1_r, f_1_r2, f_1_r3, f_fie, f_fie2, f_fie3,
           f_mend, f_mend2, f_mend3, f_z1_2, f_z1_3, f_packing_area, f_packing_area2, f_packing_area3,
           cn_1, cn_2, cn_3, sc, bc]
    """
    # 0) 입력 파라미터 유효성 검사
    if not isinstance(tol, (int, float)) or not isinstance(bottom, bool):
        return ['0. Input parameter(s) do not have correct format.'] + [None]*21

    tol = float(tol)

    # 1) slab → pymatgen Structure 객체로 변환 시도
    struc = None
    error = None

    # 1-1) 이미 Structure 타입이라면 그대로 사용
    if isinstance(slab, Structure):
        struc = slab

    # 1-2) 파일 경로에서 읽어들이기 시도
    if struc is None:
        try:
            # read() 대신 CifParser 예시. 실제 코드에 맞춰 수정 필요.
            parser = CifParser(slab)
            struc = parser.get_structures()[0]
        except Exception:
            pass

    # 1-3) ASE Atoms → Structure 변환 시도
    if struc is None:
        try:
            ase_atoms = AseAtomsAdaptor.get_atoms(slab)
            struc = AseAtomsAdaptor.get_structure(ase_atoms)
        except Exception:
            pass

    # 1-4) dict 형태 → Structure.from_dict → ASE → Structure
    if struc is None:
        try:
            tmp = Structure.from_dict(slab)
            ase_atoms = AseAtomsAdaptor.get_atoms(tmp)
            struc = AseAtomsAdaptor.get_structure(ase_atoms)
        except Exception:
            error = '1.2. Could not convert/handle input structure.'

    if struc is None and error is None:
        error = '1.0. Could not convert/handle input structure.'

    # 2) 빨리 종료할 경우
    if error:
        return [error] + [None]*21

    # 3) 허용되지 않는 원소 필터링 (비활성 가스, 방사성 등)
    forbidden = {'He','Ne','Ar','Kr','Xe','At','Rn','Fr',
                 'Cm','Bk','Cf','Es','Fm','Md','No','Lr'}
    for el in struc.symbol_set:
        if el in forbidden:
            return ['2. Structure contains element not supported for featurization.'] + [None]*21

    # 4) bc, sc 계산 (이 과정에서 Composition, featurizer를 쓰므로 메모리 크게 사용하지 않음)
    bc = band_center_featurizer.featurize(
        Composition(struc.composition.reduced_formula)
    )[0]
    sc = sc_featurizer.featurize(slab)

    # 5) 좌표, 셀 정보 추출
    #    - pos_raw: numpy 배열 뷰
    pos_raw = struc.cart_coords   # (N,3) shape, pymatgen의 내부 배열 뷰를 그대로 사용
    #    - cell lengths: a,b,c, alpha,beta,gamma
    cell = struc.lattice.lengths + struc.lattice.angles  # (a,b,c, alpha, beta, gamma)

    N_atoms = len(pos_raw)
    if N_atoms <= 3:
        return ['3. Slab less than 4 atomic layers in z-direction before applying tolerance.'] + [None]*21

    # 6) z 좌표만 분리하여 정렬(내림차순)
    #    (메모리 복제 없이도 np.sort/argsort 활용 가능)
    z_coords = pos_raw[:, 2]
    # 음수 좌표 보정: 모든 좌표가 음수인 경우 z를 양수로 뒤집음 (브로드캐스트, 뷰 복사 최소화)
    if z_coords[0] < 0 or np.all(z_coords < 0):
        # *-1 연산으로 복사본이 생기지만, 전체 pos_raw를 뒤집어도 하나만 생성
        pos_raw = pos_raw * -1
        z_coords = z_coords * -1

    # 7) z_coords 내림차순 정렬 인덱스
    #    예: 예를 들어 z_coords = [ 5.0, 3.2, 4.1, 1.0 ]
    #    args = [0,2,1,3] (가장 높은 z부터 순차)
    sorted_idx_desc = np.argsort(-z_coords)  # 내림차순 argsort

    # 8) "tol" 기준으로 층(atomic layer) 그룹화
    #    최상단 원자 집합부터 시작하여 z 차이 tol 이내에 속하는 원자들을 하나의 층으로 간주
    indices_list = []   # e.g. [[idx_층1], [idx_층2], [idx_층3], ... ]
    used = np.zeros(N_atoms, dtype=bool)  # 해당 원자가 이미 배정되었는지 마킹

    for idx in sorted_idx_desc:
        if used[idx]:
            continue
        # 이번 층의 대표 z 기준
        z_ref = z_coords[idx]
        # tol 내에 속하는 인덱스들만 필터링
        # mask = (z_coords >= z_ref - tol) & (~used)
        # 단일 비교로 메모리 복제 최소화
        layer_members = []
        for j in sorted_idx_desc:
            if used[j]:
                continue
            if z_coords[j] >= z_ref - tol:
                layer_members.append(j)
                used[j] = True
        indices_list.append(layer_members)

    # 9) 층 개수 검사 (slab이 충분한 layer를 가지는지)
    if len(indices_list) < 3:
        return [f'4. Slab less than 3 atomic layers in z-direction, with a tolerance = {tol} A.'] + [None]*21

    # 10) 최상단과 최하단 사이의 진공(vacuum) 검사
    #     - 가장 위층(첫 번째) 원자 z와 가장 아래층(마지막) 원자 z 차이가 c-방향 길이보다 충분히 작은지
    z_top = z_coords[indices_list[0][0]]
    z_bottom = z_coords[indices_list[-1][0]]
    c_length = cell[2]  # c 축 길이
    if z_top - z_bottom > c_length - 5:
        return ['6. Input structure either has no vacuum between slabs or is not oriented in z-direction.'] + [None]*21

    # 11) 각 표면(맨 위/맨 아래)과 그 다음 두 층(총 3개층) 인덱스 뽑아오기
    #     bottom=True 면 맨 아래층부터, False 면 맨 윗층부터 시작
    if bottom:
        s_indices = [indices_list[-1], indices_list[-2], indices_list[-3]]
    else:
        s_indices = [indices_list[0], indices_list[1], indices_list[2]]

    # 12) feature를 저장할 리스트 미리 할당 (메모리 오버헤드를 줄이기 위해 append만 함)
    f_chi, f_chi2, f_chi3 = [], [], []
    f_1_r, f_1_r2, f_1_r3 = [], [], []
    f_fie, f_fie2, f_fie3 = [], [], []
    f_mend, f_mend2, f_mend3 = [], [], []
    cn_1, cn_2, cn_3 = [], [], []

    chem_symbols = struc.species       # 예: ['Pd', 'Pd', 'O', ...]
    # Cell 면적 (packing area 계산 시 반복 쓰기를 방지)
    a, b, c_len, alpha, beta, gamma = cell
    sin_gamma = math.sin(math.radians(gamma))
    base_area = a * b * sin_gamma      # 단일 면적

    # 13) Layer 1 (맨 위 또는 맨 아래층)
    for atom_idx in s_indices[0]:
        el = chem_symbols[atom_idx]
        # f_chi: 전기음성도 (Element 객체 생성은 최소화)
        f_chi.append(Element(el).X)
        # cn_1: coordination number (cn_featurizer 결과)
        cn_1.append(cn_featurizer.featurize(slab, atom_idx))

        # 반지름 역수: atomic_radius_calculated이 있으면 우선 사용
        elem_obj = Element(el)
        if elem_obj.atomic_radius_calculated:
            f_1_r.append(1.0 / elem_obj.atomic_radius_calculated)
        else:
            f_1_r.append(1.0 / elem_obj.atomic_radius)

        # fiéllen (예: fie 배열에서 Z idx 사용)
        f_fie.append(fie[elem_obj.Z])
        # 멘델레브 (mendeleev 배열에서 Z idx 사용)
        f_mend.append(mendeleev[elem_obj.Z])

    count1 = len(s_indices[0])
    f_packing_area1 = count1 / base_area

    # 14) Layer 2
    #     f_z1_2: 층 1과 층 2 사이 z 차이
    z1 = z_coords[s_indices[0][0]]
    z2 = z_coords[s_indices[1][0]]
    f_z1_2 = abs(z1 - z2)

    for atom_idx in s_indices[1]:
        el = chem_symbols[atom_idx]
        f_chi2.append(Element(el).X)
        cn_2.append(cn_featurizer.featurize(slab, atom_idx))

        elem_obj = Element(el)
        if elem_obj.atomic_radius_calculated:
            f_1_r2.append(1.0 / elem_obj.atomic_radius_calculated)
        else:
            f_1_r2.append(1.0 / elem_obj.atomic_radius)

        f_fie2.append(fie[elem_obj.Z])
        f_mend2.append(mendeleev[elem_obj.Z])

    count2 = len(s_indices[1])
    f_packing_area2 = count2 / base_area

    # 15) Layer 3
    z3 = z_coords[s_indices[2][0]]
    f_z1_3 = abs(z1 - z3)

    for atom_idx in s_indices[2]:
        el = chem_symbols[atom_idx]
        f_chi3.append(Element(el).X)
        cn_3.append(cn_featurizer.featurize(slab, atom_idx))

        elem_obj = Element(el)
        if elem_obj.atomic_radius_calculated:
            f_1_r3.append(1.0 / elem_obj.atomic_radius_calculated)
        else:
            f_1_r3.append(1.0 / elem_obj.atomic_radius)

        f_fie3.append(fie[elem_obj.Z])
        f_mend3.append(mendeleev[elem_obj.Z])

    count3 = len(s_indices[2])
    f_packing_area3 = count3 / base_area

    # 16) 최종 결과 리스트로 묶어서 반환
    #     출력 순서는 원본 코드와 동일하게 맞춤
    result = [
        None,               # 에러 없음
        f_chi, f_chi2, f_chi3,
        f_1_r, f_1_r2, f_1_r3,
        f_fie, f_fie2, f_fie3,
        f_mend, f_mend2, f_mend3,
        f_z1_2, f_z1_3,
        f_packing_area1, f_packing_area2, f_packing_area3,
        cn_1, cn_2, cn_3,
        sc, bc
    ]
    return result

def raw_to_final_features(raw,
                          labels=['f_chi', 'f_chi2', 'f_chi3',
                                  'f_1_r', 'f_1_r2', 'f_1_r3',
                                  'f_fie', 'f_fie2', 'f_fie3',
                                  'f_mend', 'f_mend2', 'f_mend3',
                                  'cn_1', 'cn_2', 'cn_3', 'sc', 'bc']):
    # 삭제할 인덱스를 모으기 위한 리스트
    deleteindex = []

    # 1) 각 라벨별로 ast.literal_eval → mean, max, min, std 계산 후 raw DataFrame에 입력
    for label in labels:
        if 'chi' in label:
            for mat in raw.index:
                try:
                    lst = ast.literal_eval(str(raw.at[mat, label]))
                    raw.at[mat, label] = statistics.mean(lst)
                    raw.at[mat, label + '_max'] = max(lst)
                    raw.at[mat, label + '_min'] = min(lst)
                    raw.at[mat, label + '_std'] = np.std(lst)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

        if '1_r' in label:
            for mat in raw.index:
                try:
                    lst = ast.literal_eval(str(raw.at[mat, label]))
                    raw.at[mat, label] = statistics.mean(lst)
                    raw.at[mat, label + '_max'] = max(lst)
                    raw.at[mat, label + '_min'] = min(lst)
                    raw.at[mat, label + '_std'] = np.std(lst)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

        if 'fie' in label:
            for mat in raw.index:
                try:
                    lst = ast.literal_eval(str(raw.at[mat, label]))
                    raw.at[mat, label] = statistics.mean(lst)
                    raw.at[mat, label + '_max'] = max(lst)
                    raw.at[mat, label + '_min'] = min(lst)
                    raw.at[mat, label + '_std'] = np.std(lst)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

        if 'mend' in label:
            for mat in raw.index:
                try:
                    lst = ast.literal_eval(str(raw.at[mat, label]))
                    raw.at[mat, label] = statistics.mean(lst)
                    raw.at[mat, label + '_max'] = max(lst)
                    raw.at[mat, label + '_min'] = min(lst)
                    raw.at[mat, label + '_std'] = np.std(lst)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

        if "cn" in label:
            for mat in raw.index:
                try:
                    lst = ast.literal_eval(str(raw.at[mat, label]))
                    flattened_list = [x for sublist in lst for x in sublist]
                    raw.at[mat, label] = statistics.mean(flattened_list)
                    raw.at[mat, label + '_max'] = max(flattened_list)
                    raw.at[mat, label + '_min'] = min(flattened_list)
                    raw.at[mat, label + '_std'] = np.std(flattened_list)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

        if "sc" in label:
            for mat in raw.index:
                try:
                    lst = raw.at[mat, label]
                    raw.at[mat, label] = np.mean(lst)
                    raw.at[mat, label + '_max'] = max(lst)
                    raw.at[mat, label + '_min'] = min(lst)
                    raw.at[mat, label + '_std'] = np.std(lst)
                except:
                    if mat not in deleteindex:
                        deleteindex.append(mat)

    # AST 오류로 표시된 행 개수 출력
    print('ast errors = ' + str(len(deleteindex)))

    # 오류가 난 행 드롭 후 인덱스 리셋
    raw = raw.drop(deleteindex)
    raw = raw.reset_index(drop=True)

    # 이후 중복 행 검사용 deleteindex 초기화
    deleteindex = []

    # -------------------------------------------------------------------
    # 2) 여기서 'chi_diff' 컬럼을 추가
    #
    #   각 행(row)에 대해
    #     f_chi_max, f_chi2_max, f_chi3_max 중 최대값
    #     f_chi_min, f_chi2_min, f_chi3_min 중 최소값
    #   을 구한 뒤, 두 값의 차이를 chi_diff라 칭함.
    #
    #   Pandas의 벡터 연산을 이용하면 한 줄로 계산이 가능합니다.
    # -------------------------------------------------------------------

    # 먼저 필요한 칼럼들이 존재하는지 확인 (만약, 일부 row에 값이 없을 수 있으므로)
    # 존재하지 않는다면 그냥 0으로 채우거나 NaN 처리해도 무방
    for col in ['f_chi_max', 'f_chi2_max', 'f_chi3_max',
                'f_chi_min', 'f_chi2_min', 'f_chi3_min']:
        if col not in raw.columns:
            # 해당 칼럼이 없으면 0을 기본값으로 추가
            raw[col] = 0.0

    # chi_max들 중 행별 최대값, chi_min들 중 행별 최소값 계산
    chi_max_cols = ['f_chi_max', 'f_chi2_max', 'f_chi3_max']
    chi_min_cols = ['f_chi_min', 'f_chi2_min', 'f_chi3_min']

    # 행별 최대값·최소값 Series 생성
    chi_maximums = raw[chi_max_cols].max(axis=1)  # 각 행마다 3개 값 중 최대
    chi_minimums = raw[chi_min_cols].min(axis=1)  # 각 행마다 3개 값 중 최소

    # 차이를 구해서 새로운 컬럼 추가
    raw['chi_diff'] = chi_maximums - chi_minimums

    # -------------------------------------------------------------------
    # 3) 중복 행 제거 로직 (원본과 동일하게 유지하되, threshold를 30으로 변경)
    # -------------------------------------------------------------------
    nbottom = 0
    for i in raw.index:
        for j in raw.index:
            # j가 i보다 크고, 인덱스 차이가 30 미만인 경우만 비교
            if j > i and j < i + 10:
                # 'f_chi' ~ 'bc' 칼럼 구간 전체 값이 거의 같은지 검사
                if (np.isclose(raw.loc[i, 'f_chi':'bc'].astype(np.double),
                               raw.loc[j, 'f_chi':'bc'].astype(np.double))).all():
                    if i not in deleteindex:
                        if raw.at[i, 'bottom']:
                            nbottom += 1
                        deleteindex.append(i)

    print('Total deleteindex = ' + str(len(deleteindex)))
    print('Bottom deleted = ' + str(nbottom))

    # 발견된 중복 행 삭제 후 인덱스 재설정
    raw = raw.drop(deleteindex)
    raw = raw.reset_index(drop=True)

    # 고유 ID 생성
    id = str(time.time())

    return raw, id, str(len(deleteindex))

"""
def featurization(slab, bottom = False, tol = 0.7):
    error = None
    if not isinstance(tol, (int, float)) or not isinstance(bottom, bool):
        error = '0.Input parameter(s) do not have correct format.'
    
    if not error:
        tol = float(tol)
        if isinstance(slab, Structure):
          print("Is Strucutre")
        else:
          print("Not Structure")
        try:
            struc = read(slab)
        except:
            error = '1.0. Could not convert/handle input structure'
        if error:
            try:
                struc = AseAtomsAdaptor.get_atoms(slab)
                error = None
            except:
                error = '1.1.Could not convert/handle input structure'
        if error:
            try:
                slabdic = Structure.from_dict(slab)
                struc = AseAtomsAdaptor.get_atoms(slabdic)
                error = None
            except:
                error = '1.2.Could not convert/handle input structure'

    if not error:
        for el in struc.get_chemical_symbols():
            if el in ['He','Ne','Ar','Kr','Xe','At','Rn','Fr','Cm','Bk','Cf','Es','Fm','Md','No','Lr']:
                error = '2.Structure contains element not supported for featurization.'
    bc = band_center_featurizer.featurize(Composition(struc.get_chemical_formula('hill')))[0]
    sc = sc_featurizer.featurize(slab)

    if not error:
        #struc *= (2,2,1)
        pos = struc.get_positions()
        if len(pos) > 3:
            if pos[0][2] < 0:#correct weird cif import; sometimes all coordinates are negative
                pos = pos * -1

            #Create list of indices from highest z position to lowest
            #--------------------------------------------------------
            counter = 0
            indices_list = []
            while counter < len(pos):
                #Find index for atom(s) with highest z-position
                surface = max(pos[:,2])
                highest_indices = []
                for ind, p in enumerate(pos):
                    if p[2] > surface - tol:
                        highest_indices.append(ind)
                #Once the index/indices of highest atom(s) is/are found, set that position to zero for the next while loop iteration
                #and increase counter by the number of highest found indices.
                if len(highest_indices) > 0:
                    indices_list.append(highest_indices)
                    for ind in highest_indices:
                        pos[ind]=[0, 0, 0]
                    counter = counter + len(highest_indices)
                else:
                    error = '5.Error. No highest index found. Counter = ' + str(counter)
                    break

            #Check there are at least 6 layers, given tolerance to group layers
            if len(indices_list) < 3 and not error:
                error = '4.Slab less than 3 atomic layers in z-direction, with a tolerance = ' + str(tol) + ' A.'

            pos = struc.get_positions()
            if pos[0][2] < 0:#correct weird cif import; sometimes all coordinates are negative
                pos = pos * -1

            #Check if structure is of form slab with vacuum in z-direction
            if pos[indices_list[0][0]][2] - pos[indices_list[-1][0]][2] > struc.get_cell_lengths_and_angles()[2] - 5:
                error = '6.Input structure either has no vacuum between slabs or is not oriented in z-direction'
        else:
            error = '3.Slab less than 4 atomic layers in z-direction before applying tolerance.'

    if not error:
        #Add features
        #------------
        chem = struc.get_chemical_symbols()
        cell = struc.get_cell_lengths_and_angles()

        #Refer to top or bottom surface index:
        sindex = -1 if bottom else 0
        sindex2 = -2 if bottom else 1
        sindex3 = -3 if bottom else 2

        #Feature Layer 1
        f_chi = []
        f_1_r = []
        f_fie = []
        f_mend = []
        cn_1 = []

        for ind in range(len(indices_list[sindex])):
            f_chi.append(Element(chem[indices_list[sindex][ind]]).X)
            cn_1.append(cn_featurizer.featurize(slab, indices_list[sindex][ind]))
            if Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius_calculated)
            else:
                f_1_r.append(1 / Element(chem[indices_list[sindex][ind]]).atomic_radius)
            f_fie.append(fie[Element(chem[indices_list[sindex][ind]]).Z])
            f_mend.append(mendeleev[Element(chem[indices_list[sindex][ind]]).Z])
        f_packing_area = len(indices_list[sindex]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        #Features layer 2
        f_z1_2 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex2][0]][2])
        f_chi2 = []
        f_1_r2 = []
        f_fie2 = []
        f_mend2 = []
        cn_2 = []

        for ind2 in range(len(indices_list[sindex2])):
            f_chi2.append(Element(chem[indices_list[sindex2][ind2]]).X)
            cn_2.append(cn_featurizer.featurize(slab, indices_list[sindex2][ind2]))
            if Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius_calculated)
            else:
                f_1_r2.append(1 / Element(chem[indices_list[sindex2][ind2]]).atomic_radius)
            f_fie2.append(fie[Element(chem[indices_list[sindex2][ind2]]).Z])
            f_mend2.append(mendeleev[Element(chem[indices_list[sindex2][ind2]]).Z])
        f_packing_area2 = len(indices_list[sindex2]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        #Features layer 3
        f_z1_3 = abs(pos[indices_list[sindex][0]][2] - pos[indices_list[sindex3][0]][2])
        f_chi3 = []
        f_1_r3 = []
        f_fie3 = []
        f_mend3 = []
        cn_3 = []

        for ind3 in range(len(indices_list[sindex3])):
            f_chi3.append(Element(chem[indices_list[sindex3][ind3]]).X)
            cn_3.append(cn_featurizer.featurize(slab, indices_list[sindex3][ind3]))
            if Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius_calculated)
            else:
                f_1_r3.append(1 / Element(chem[indices_list[sindex3][ind3]]).atomic_radius)
            f_fie3.append(fie[Element(chem[indices_list[sindex3][ind3]]).Z])
            f_mend3.append(mendeleev[Element(chem[indices_list[sindex3][ind3]]).Z])
        f_packing_area3 = len(indices_list[sindex3]) / (cell[0] * cell[1] * math.sin(math.radians(cell[5])))

        return [error, f_chi, f_chi2, f_chi3, f_1_r, f_1_r2, f_1_r3, f_fie, f_fie2, f_fie3,
            f_mend, f_mend2, f_mend3, f_z1_2, f_z1_3, f_packing_area, f_packing_area2, f_packing_area3, cn_1, cn_2, cn_3, sc, bc]
    else:
        return [error, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None, None, None]


def raw_to_final_features(raw, labels = ['f_chi', 'f_chi2', 'f_chi3', 'f_1_r', 'f_1_r2',
                                    'f_1_r3', 'f_fie', 'f_fie2', 'f_fie3', 'f_mend', 'f_mend2', 'f_mend3', 'cn_1', 'cn_2', 'cn_3', 'sc' , 'bc']):
        deleteindex = []
        for label in labels:
            if 'chi' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                        raw.at[mat, label + '_std'] = np.std(list)

                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if '1_r' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                        raw.at[mat, label + '_std'] = np.std(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if 'fie' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                        raw.at[mat, label + '_std'] = np.std(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)
            if 'mend' in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        raw.at[mat, label] = statistics.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                        raw.at[mat, label + '_std'] = np.std(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)

            if "cn" in label:
                for mat in raw.index:
                    try:
                        list = ast.literal_eval(str(raw.at[mat, label]))
                        flattened_list = [x for sublist in list for x in sublist]
                        raw.at[mat, label] = statistics.mean(flattened_list)
                        raw.at[mat, label + '_max'] = max(flattened_list)
                        raw.at[mat, label + '_min'] = min(flattened_list)
                        raw.at[mat, label + '_std'] = np.std(flattened_list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)

            if "sc" in label:
                for mat in raw.index:
                    try:
                        list = raw.at[mat,label]
                        raw.at[mat, label] = np.mean(list)
                        raw.at[mat, label + '_max'] = max(list)
                        raw.at[mat, label + '_min'] = min(list)
                        raw.at[mat, label + '_std'] = np.std(list)
                    except:
                        if mat not in deleteindex:
                            deleteindex.append(mat)

        print('ast errors = ' + str(len(deleteindex)))
        raw = raw.drop(deleteindex)
        raw = raw.reset_index(drop=True)
        deleteindex = []
        nbottom = 0
        for i in raw.index:
            for j in raw.index:
                if j > i and j < i + 30:
                    if (np.isclose(raw.loc[i,'f_chi':'bc'].astype(np.double), raw.loc[j,'f_chi':'bc'].astype(np.double))).all():
                        if i not in deleteindex:
                            if raw.at[i, 'bottom']:
                                nbottom += 1
                            deleteindex.append(i)
        print('Total deleteindex = ' + str(len(deleteindex)))
        print('Bottom deleted = ' +str(nbottom))
        raw = raw.drop(deleteindex)
        raw = raw.reset_index(drop=True)

        id = str(time.time())
        return raw, id, str(len(deleteindex))
"""

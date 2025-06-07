
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
#from matminer.featurizers.structure import StructuralComplexity
from matminer.featurizers.composition.element import BandCenter
from matminer.featurizers.composition.orbital import ValenceOrbital
from pymatgen.core import Composition

cn_featurizer = CoordinationNumber.from_preset('VoronoiNN')
band_center_featurizer = BandCenter()
vo_featurizer = ValenceOrbital()
#sc_featurizer = StructuralComplexity(symprec=0.1)

with open('atomic_features/firstionizationenergy.txt') as f:
    content = f.readlines()
fie = [float(x.strip()) for x in content]

with open('atomic_features/mendeleev.txt') as f:
    content = f.readlines()
mendeleev = [float(x.strip()) for x in content]

def featurization(slab, tol=0.7, ads_sites=None, cutoff=3.0):
    # 0) 입력 파라미터 검증 (생략)

    # 1) slab → Structure 변환 (최소한의 try/except로 간략화)
    struc = None
    if isinstance(slab, Structure):
        struc = slab
    else:
        try:
            parser = CifParser(slab)
            struc = parser.get_structures()[0]
        except:
            try:
                ase_atoms = AseAtomsAdaptor.get_atoms(slab)
                struc = AseAtomsAdaptor.get_structure(ase_atoms)
            except:
                return ['1.0. Could not convert/handle input structure.'] + [None]*25

    # 2) 금지 원소 검사 (생략)
    forbidden = {...}
    unique_elems = set(struc.species)
    if unique_elems & forbidden:
        return ['2. Structure contains element not supported.'] + [None]*25

    # 3) 밴드 중심 계산 (생략)
    formula = struc.composition.reduced_formula
    bc = band_center_featurizer.featurize(Composition(formula))[0]

    # 4) 좌표, 셀 정보 가져오기 (생략)
    pos_raw = struc.cart_coords
    z_coords = pos_raw[:, 2]
    if np.all(z_coords < 0):
        pos_raw = -pos_raw
        z_coords = -z_coords

    # 5) 층 그룹화 (생략)
    N_atoms = len(z_coords)
    if N_atoms <= 3:
        return ['3. Slab less than 4 atomic layers'] + [None]*25

    sorted_idx_desc = np.argsort(-z_coords)
    used = np.zeros(N_atoms, dtype=bool)
    indices_list = []
    for idx in sorted_idx_desc:
        if used[idx]: continue
        z_ref = z_coords[idx]
        mask_layer = (~used) & (z_coords >= z_ref - tol)
        layer_members = np.nonzero(mask_layer)[0]
        used[layer_members] = True
        indices_list.append(layer_members.tolist())

    # 6) 층 개수 검사
    if len(indices_list) < 3:
        return [f'4. Layers less than 3 with tol={tol}'] + [None]*25

    # 7) 진공 체크 (생략)

    # 8) Layer 인덱스 추출
    s_indices = [indices_list[0], indices_list[1], indices_list[2]]

    # 9) Element 속성 캐싱
    elem_cache = {}
    for el in set(struc.species):
        eobj = Element(el)
        elem_cache[el] = {
            'X': eobj.X,
            'Z': eobj.Z,
            've': np.sum(vo_featurizer.featurize(Composition(str(el)))[0:4]),
            'radius_inv': 1.0 / (eobj.atomic_radius_calculated 
                                 or eobj.atomic_radius)
        }

    # 10) Cell 면적 계산 (생략)
    a, b, c_len, alpha, beta, gamma = struc.lattice.lengths + struc.lattice.angles
    base_area = a * b * math.sin(math.radians(gamma))

    # 11) Layer별 Feature 추출을 “리스트 컴프리헨션 + 캐시”로 대체
    # -- Layer 1 --
    layer0_idx = np.array(s_indices[0], dtype=int)
    layer0_elems = [struc.species[i] for i in layer0_idx]

    f_chi   = [elem_cache[el]['X'] for el in layer0_elems]
    f_1_r   = [elem_cache[el]['radius_inv'] for el in layer0_elems]
    f_fie   = [fie[elem_cache[el]['Z']] for el in layer0_elems]
    f_mend  = [mendeleev[elem_cache[el]['Z']] for el in layer0_elems]
    f_ve = [elem_cache[el]['ve'] for el in layer0_elems]
    cn_1    = [cn_featurizer.featurize(slab, idx) for idx in layer0_idx]

    count1  = layer0_idx.size
    f_packing_area1 = count1 / base_area

    # -- Layer 2 --
    z1 = z_coords[layer0_idx[0]]
    layer1_idx = np.array(s_indices[1], dtype=int)
    layer1_elems = [struc.species[i] for i in layer1_idx]

    f_chi2  = [elem_cache[el]['X'] for el in layer1_elems]
    f_1_r2  = [elem_cache[el]['radius_inv'] for el in layer1_elems]
    f_fie2  = [fie[elem_cache[el]['Z']] for el in layer1_elems]
    f_mend2 = [mendeleev[elem_cache[el]['Z']] for el in layer1_elems]
    f_ve2 = [elem_cache[el]['ve'] for el in layer1_elems]
    #cn_2    = [cn_featurizer.featurize(slab, idx) for idx in layer1_idx]

    count2  = layer1_idx.size
    f_packing_area2 = count2 / base_area
    z2 = z_coords[layer1_idx[0]]
    f_z1_2 = abs(z1 - z2)

    # -- Layer 3 --
    layer2_idx = np.array(s_indices[2], dtype=int)
    layer2_elems = [struc.species[i] for i in layer2_idx]

    f_chi3  = [elem_cache[el]['X'] for el in layer2_elems]
    f_1_r3  = [elem_cache[el]['radius_inv'] for el in layer2_elems]
    f_fie3  = [fie[elem_cache[el]['Z']] for el in layer2_elems]
    f_mend3 = [mendeleev[elem_cache[el]['Z']] for el in layer2_elems]
    f_ve3 = [elem_cache[el]['ve'] for el in layer2_elems]
    #cn_3    = [cn_featurizer.featurize(slab, idx) for idx in layer2_idx]

    count3  = layer2_idx.size
    f_packing_area3 = count3 / base_area
    z3 = z_coords[layer2_idx[0]]
    f_z1_3 = abs(z1 - z3)

    # 12) Adsorption Site 주변 원자 특성 계산
    ads_chi = []
    ads_fie = []
    ads_ve  = []
    ads_cn  = []

    if ads_sites:
        for coord in ads_sites:
            # radius 반경 내 원자 검색
            neighbors = struc.get_sites_in_sphere(coord, cutoff)  # [(site, dist), ...]
            if not neighbors:
                ads_chi.append(None)
                ads_fie.append(None)
                ads_ve.append(None)
                ads_cn.append(None)
                continue

            chis, fies, ves, cns = [], [], [], []
            for site, dist in neighbors:
                el = str(site.specie)
                chis.append(elem_cache[el]['X'])
                fies.append(fie[elem_cache[el]['Z']])
                ves.append(elem_cache[el]['ve'])
                # 배위수 계산을 위해 구조 내 인덱스 추정
                # (가장 가까운 원자 인덱스 찾기)
                idx = int(np.argmin(np.linalg.norm(pos_raw - site.coords, axis=1)))
                cns.append(cn_featurizer.featurize(slab, idx))

            ads_chi.append(np.mean(chis))
            ads_fie.append(np.mean(fies))
            ads_ve.append(np.mean(ves))
            ads_cn.append(np.mean(cns))
    else:
        # ads_sites 정보 없을 때
        ads_chi = ads_fie = ads_ve = ads_cn = [None]

    # 13) 최종 결과 리스트에 추가
    result = [
        None, 
        # 기존 0~11단계에서 생성된 모든 피처들...
        f_chi, f_chi2, f_chi3,
        f_1_r, f_1_r2, f_1_r3,
        f_fie, f_fie2, f_fie3,
        f_mend, f_mend2, f_mend3,
        f_ve, f_ve2, f_ve3,
        f_z1_2, f_z1_3,
        f_packing_area1, f_packing_area2, f_packing_area3,
        cn_1,
        bc,
        # 이제 추가된 adsorption site 피처
        ads_chi, ads_fie, ads_ve, ads_cn
    ]
    return result

def raw_to_final_features(raw,
                          labels=['f_chi', 'f_chi2', 'f_chi3',
    'f_1_r', 'f_1_r2', 'f_1_r3',
    'f_fie', 'f_fie2', 'f_fie3',
    'f_mend', 'f_mend2', 'f_mend3',
    'f_ve', 'f_ve2', 'f_ve3',
    'cn_1']):
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

        if 've' in label:
            for mat in raw.index:
                try:
                    lst = raw.at[mat, label]
                    raw.at[mat, label] = statistics.mean(lst)
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
    for i in raw.index:
        for j in raw.index:
            # j가 i보다 크고, 인덱스 차이가 30 미만인 경우만 비교
            if j > i and j < i + 5:
                # 'f_chi' ~ 'bc' 칼럼 구간 전체 값이 거의 같은지 검사
                if (np.isclose(raw.loc[i, 'f_chi':'cn_1_std'].astype(np.double),
                               raw.loc[j, 'f_chi':'cn_1_std'].astype(np.double))).all():
                    if i not in deleteindex:
                        deleteindex.append(i)

    print('Total deleteindex = ' + str(len(deleteindex)))

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

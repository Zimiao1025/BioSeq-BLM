import sys
from ..utils.utils_const import DNA, RNA, PROTEIN
from .acc import acc, auto_correlation, pdt, nd
from .pse import zcpseknc
from .profile import make_acc_pssm_vector, pssm_dt_method, pssm_rt_method, pdt_profile
from .motif_pssm import motif_pssm
from ..utils.utils_write import vectors2files

METHODS_ACC_P = ['DAC', 'DCC', 'DACC', 'TAC', 'TCC', 'TACC', 'MAC', 'GAC', 'NMBAC', 'AC', 'CC', 'ACC', 'PDT']
METHODS_ACC_S = ['DAC', 'DCC', 'DACC', 'TAC', 'TCC', 'TACC', 'AC', 'CC', 'ACC']
METHODS_AC = ['DAC', 'TAC', 'AC']
METHODS_CC = ['DCC', 'TCC', 'CC']
METHODS_ACC = ['DACC', 'TACC', 'ACC']

K_2_DNA_METHODS = ['DAC', 'DCC', 'DACC']
K_3_DNA_METHODS = ['TAC', 'TCC', 'TACC']

DI_IND_6_DNA = ['Rise', 'Roll', 'Shift', 'Slide', 'Tilt', 'Twist']
TRI_IND_DNA = ['Dnase I', 'Bendability (DNAse)']

DI_IND_RNA = ['Rise (RNA)', 'Roll (RNA)', 'Shift (RNA)', 'Slide (RNA)', 'Tilt (RNA)', 'Twist (RNA)']
IND_3_PROTEIN = ['Hydrophobicity', 'Hydrophilicity', 'Mass']

ALL_DI_DNA_IND = ['Base stacking', 'Protein induced deformability', 'B-DNA twist',
                  'Dinucleotide GC Content', 'A-philicity', 'Propeller twist',
                  'Duplex stability-free energy', 'Duplex stability-disrupt energy', 'DNA denaturation',
                  'Bending stiffness', 'Protein DNA twist', 'Stabilising energy of Z-DNA',
                  'Aida_BA_transition', 'Breslauer_dG', 'Breslauer_dH', 'Breslauer_dS',
                  'Electron_interaction', 'Hartman_trans_free_energy', 'Helix-Coil_transition',
                  'Ivanov_BA_transition', 'Lisser_BZ_transition', 'Polar_interaction', 'SantaLucia_dG',
                  'SantaLucia_dH', 'SantaLucia_dS', 'Sarai_flexibility', 'Stability', 'Stacking_energy',
                  'Sugimoto_dG', 'Sugimoto_dH', 'Sugimoto_dS', 'Watson-Crick_interaction', 'Twist',
                  'Tilt', 'Roll', 'Shift', 'Slide', 'Rise', 'Stacking energy', 'Bend', 'Tip',
                  'Inclination', 'Major Groove Width', 'Major Groove Depth', 'Major Groove Size',
                  'Major Groove Distance', 'Minor Groove Width', 'Minor Groove Depth',
                  'Minor Groove Size', 'Minor Groove Distance', 'Persistance Length',
                  'Melting Temperature', 'Mobility to bend towards major groove',
                  'Mobility to bend towards minor groove', 'Propeller Twist', 'Clash Strength',
                  'Enthalpy', 'Free energy', 'Twist_twist', 'Tilt_tilt', 'Roll_roll', 'Twist_tilt',
                  'Twist_roll', 'Tilt_roll', 'Shift_shift', 'Slide_slide', 'Rise_rise', 'Shift_slide',
                  'Shift_rise', 'Slide_rise', 'Twist_shift', 'Twist_slide', 'Twist_rise', 'Tilt_shift',
                  'Tilt_slide', 'Tilt_rise', 'Roll_shift', 'Roll_slide', 'Roll_rise', 'Slide stiffness',
                  'Shift stiffness', 'Roll stiffness', 'Rise stiffness', 'Tilt stiffness',
                  'Twist stiffness', 'Wedge', 'Direction', 'Flexibility_slide', 'Flexibility_shift',
                  'Entropy']
DEFAULT_DI_DNA_IND = ['Twist', 'Tilt', 'Roll', 'Shift', 'Slide', 'Rise']
ALL_TRI_DNA_IND = ['Bendability-DNAse', 'Bendability-consensus', 'Trinucleotide GC Content',
                   'Nucleosome positioning', 'Consensus_roll', 'Consensus_Rigid', 'Dnase I',
                   'Dnase I-Rigid', 'MW-Daltons', 'MW-kg', 'Nucleosome', 'Nucleosome-Rigid']
DEFAULT_TRI_DNA_IND = ['Nucleosome positioning', 'Dnase I']
ALL_RNA_IND = ['Shift', 'Slide', 'Rise', 'Tilt', 'Roll', 'Twist', 'Stacking energy', 'Enthalpy', 'Entropy',
               'Free energy', 'Hydrophilicity']
DEFAULT_RNA_IND = ['Shift', 'Slide', 'Rise', 'Tilt', 'Roll', 'Twist']


def read_k(alphabet, _method, k):
    if alphabet == 'Protein':
        return 1
    elif alphabet == 'RNA':
        return 2

    if _method in K_2_DNA_METHODS:
        return 2
    elif _method in K_3_DNA_METHODS:
        return 3
    elif _method == 'ZCPseKNC':
        return k
    else:
        print("Error in read_k.")


def read_index(index_file):
    with open(index_file) as f_ind:
        lines = f_ind.readlines()
        ind_list = [index.rstrip() for index in lines]
        return ind_list


def syntax_rules(method, input_file, category, sample_num_list, out_format, out_file_list, cur_dir, args, **param_dict):
    res = None
    sw_dir = cur_dir + '/software/'
    if category == 'DNA':
        alphabet = DNA
    elif category == 'RNA':
        alphabet = RNA
    else:
        alphabet = PROTEIN

    if method in METHODS_ACC_S:
        with open(input_file) as f:
            k = read_k(category, method, 0)

            # Get index_list.
            if args.pp_file is not None:
                ind_list = read_index(args.pp_file)
                # print(ind_list)
            else:
                ind_list = []

            default_e = []
            # Set default pp index_list.
            if category == 'DNA':
                if k == 2:
                    default_e = DI_IND_6_DNA
                elif k == 3:
                    default_e = TRI_IND_DNA
            elif category == 'RNA':
                default_e = DI_IND_RNA
            else:
                default_e = IND_3_PROTEIN

            if method in METHODS_AC:
                theta_type = 1
            elif method in METHODS_CC:
                theta_type = 2
            else:
                theta_type = 3

            if args.ui_file is None and len(ind_list) == 0 and args.a is False:
                lag = param_dict['lag']
                res = acc(f, k, lag, default_e, alphabet,
                          extra_index_file=args.ui_file, all_prop=args.a, theta_type=theta_type)
            else:
                lag = param_dict['lag']
                res = acc(f, k, lag, ind_list, alphabet,
                          extra_index_file=args.ui_file, all_prop=args.a, theta_type=theta_type)

    elif method in ['MAC', 'GAC', 'NMBAC']:
        lamada = param_dict['lamada']
        assert 0 < lamada < 16, 'The value of -lamada should be larger than 0 and smaller than 16.'

        if args.a is None:
            args.a = False
        if category == 'DNA':
            if args.oli == 0:
                if args.a is True:
                    res = auto_correlation(method, input_file, props=ALL_DI_DNA_IND, k=2,
                                           lamada=lamada, alphabet=alphabet)

                else:
                    res = auto_correlation(method, input_file, props=DEFAULT_DI_DNA_IND, k=2,
                                           lamada=lamada, alphabet=alphabet)

            if args.oli == 1:
                if args.a is True:
                    res = auto_correlation(method, input_file, props=ALL_TRI_DNA_IND, k=3,
                                           lamada=lamada, alphabet=alphabet)

                else:
                    res = auto_correlation(method, input_file, props=DEFAULT_TRI_DNA_IND, k=3,
                                           lamada=lamada, alphabet=alphabet)

        elif category == 'RNA':
            if args.a is True:
                res = auto_correlation(method, input_file, props=ALL_RNA_IND, k=2,
                                       lamada=lamada, alphabet=alphabet)
            else:
                res = auto_correlation(method, input_file, props=DEFAULT_RNA_IND, k=2,
                                       lamada=lamada, alphabet=alphabet)
        else:
            error_info = "'MAC', 'GAC', 'NMBAC' method only for RNA and DNA sequence, please read manual"
            sys.stderr.write(error_info)
            return False

    elif method == 'PDT':
        lamada = param_dict['lamada']
        assert 0 < lamada < 16, 'The value of -lamada should be larger than 0 and smaller than 16.'

        res = pdt(input_file, lamada, sw_dir)

    elif method == 'ZCPseKNC':
        lamada = param_dict['lamada']
        assert 0 < lamada < 16, 'The value of -lamada should be larger than 0 and smaller than 16.'
        res = zcpseknc(input_file, k=param_dict['k'], w=param_dict['w'], lamada=lamada, alphabet=DNA)
    elif method in ['ACC-PSSM', 'AC-PSSM', 'CC-PSSM']:
        if method == 'ACC-PSSM':
            vec_type = 'acc'
        elif method == 'AC-PSSM':
            vec_type = 'ac'
        else:
            vec_type = 'cc'
        lag = param_dict['lag']
        if lag < 1:
            print('The value of -lag should be larger than 0.')
            return False
        else:
            res = make_acc_pssm_vector(input_file, lag, vec_type, sw_dir, process_num=param_dict['cpu'])
    elif method == 'ND':
        res = nd(input_file, alphabet, fixed_len=args.fixed_len)

    elif method == 'PSSM-DT':
        res = pssm_dt_method(input_file, param_dict['cpu'], sw_dir)

    elif method == 'PSSM-RT':
        res = pssm_rt_method(input_file, param_dict['cpu'], sw_dir, fixed_len=args.fixed_len)
    elif method == 'PDT-Profile':
        lamada = param_dict['lamada']
        assert 0 < lamada < 16, 'The value of -lamada should be larger than 0 and smaller than 16.'

        res = pdt_profile(input_file, param_dict['n'], lamada, sw_dir, process_num=param_dict['cpu'])
    elif method == 'Motif-PSSM':
        # 这里的all_data应该为PSSM矩阵，后续需要修改

        res = motif_pssm(input_file, PROTEIN, process_num=param_dict['cpu'],
                         batch_size=param_dict['batch_size'],
                         motif_file=args.motif_file, motif_database=args.motif_database,
                         fixed_len=args.fixed_len, cur_dir=cur_dir)
    else:
        error_info = 'The method of syntax rules is wrong, please check!'
        sys.stderr.write(error_info)
        return False

    vectors2files(res, sample_num_list, out_format, out_file_list)

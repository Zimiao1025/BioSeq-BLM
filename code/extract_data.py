import os.path
import sys
t = sys.argv[1]
c = '../results/batch/' + str(t) + '/'

for a in os.walk(c):
    for dirname in a[1]:
        if os.path.exists(c + dirname + '/param.txt'):
            if os.path.exists(c + dirname + '/cv_model.model'):
                with open(c + dirname + '/final_results.txt') as f, open(c + dirname + '/cv_model.model') as m,\
                        open(c + dirname + '/param.txt') as p, open(c + 'results.txt', 'a') as w:
                    lines = f.readlines()
                    m_lines = m.readlines()
                    p_lines = p.readlines()
                    for line in lines[1:6]:
                        dirname += ' ' + line.strip().split(' ')[-1]
                    if t.split('/')[1] == 'svm':
                        param = m_lines[0].strip().split(',')
                        dirname += ' ' + param[1] + ' ' + param[3]
                    else:
                        param = m_lines[-1].strip().split(',')
                        dirname += ' ' + param[0]
                    dirname += ' ' + p_lines[0]
                    w.writelines(dirname + '\n')
            else:
                with open(c + dirname + '/cv_eval_results.txt') as f, open(c + dirname + '/param.txt') as p, \
                        open(c + 'results.txt', 'a') as w:
                    lines = f.readlines()
                    p_lines = p.readlines()
                    for line in lines[1:6]:
                        dirname += ' ' + line.strip().split(' ')[-1]
                    if t.split('/')[1] == 'rf':
                        dirname += ' ' + 'None'
                    dirname += ' ' + p_lines[0]
                    w.writelines(dirname + '\n')
        else:
            with open(c + dirname + '/cv_eval_results.txt') as f, open(c + dirname + '/cv_model.model') as m,\
                    open(c + 'results.txt', 'a') as w:
                lines = f.readlines()
                m_lines = m.readlines()
                for line in lines[1:6]:
                    dirname += ' ' + line.strip().split(' ')[-1]
                if t.split('/')[1] == 'svm':
                    param = m_lines[0].strip().split(',')
                    dirname += ' ' + param[1] + ' ' + param[3]
                else:
                    param = m_lines[-1].strip().split(',')
                    dirname += ' ' + param[0]
                dirname += ' ' + 'None'
                w.writelines(dirname + '\n')



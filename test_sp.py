import subprocess

mt5_proc = subprocess.Popen("OrtMT5.exe", stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)

in_str = '''17
1042
264
462
338
12001
25134
31150
70694
10978
67916
81634
574
86977
56157
66637
309
1
'''
mt5_proc.stdin.write(in_str)
od1 = mt5_proc.stdout.readline()
print(od1)
mt5_proc.stdin.write(in_str)
od2 = mt5_proc.stdout.readline()
print(od2)
mt5_proc.communicate("exit")

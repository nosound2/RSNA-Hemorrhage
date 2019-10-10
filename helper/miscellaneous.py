import torch
import smtplib

def device_by_name(name):
    assert torch.cuda.is_available(),"No cuda device"
    device=None
    for i in range(torch.cuda.device_count()):
        dv=torch.device("cuda:{}".format(i))
        if name in torch.cuda.get_device_name(dv):
            device=dv
            break
    assert device, "device {} not found".format(name)
    return device

def gmean_torch(t,w,op = None):
    wn = w/w.sum()
    tr = t.unsqueeze(-1).transpose(0,-1).squeeze(0)
    if op is not None:
        tr=op(tr)
    tt=torch.ones_like(tr[...,0])
    for i in range(tr.shape[-1]):
        tt=tt*torch.pow(tr[...,i],wn[i])
        
class Email_Progress():
    def __init__(self,source_email,source_password,target_email,title):
        self.source_email=source_email
        self.source_password=source_password
        self.target_email=target_email
        self.title=title
        
    def __call__(self,history):
        str_list=['Subject:{}\n'.format(self.title)]+[str(d)[1:-2].replace("'",'').replace(':','')+'\n' for d in history]
        email_text=''.join(str_list)            
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(self.source_email, self.source_password)
            server.sendmail(self.source_email, self.target_email, email_text)
            server.close()
            return 0
        except Exception as e:
            print(e)
            print ('Something went wrong...')
            return e
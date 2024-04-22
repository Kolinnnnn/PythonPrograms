import numpy as np
import matplotlib.pyplot as plt

def generujsygnal(strumien,tb, a1, a2, W, B):
    tbit = np.linspace(0, tb, tc*fs)
    t = np.linspace(0, tb*B, B*tc*fs)
    fn1 = (W+1)/tb
    fn2 = (W+2)/tb
    ask = np.zeros(len(t))
    psk = np.zeros(len(t))
    fsk = np.zeros(len(t))
    nosny = np.zeros(len(t))
    for i in range(B):
        bit = int(strumien[i])
        if bit == 0:
            zat = a1*np.sin(2*np.pi*fn*tbit)
            zpt = np.sin(2*np.pi*fn*tbit)
            zft = np.sin(2*np.pi*fn1*tbit)
            nosn = a1*np.sin(2*np.pi*fn*tbit)
        elif bit == 1:
            zat = a2*np.sin(2*np.pi*fn*tbit)
            zpt = np.sin(2*np.pi*fn*tbit+np.pi)
            zft = np.sin(2*np.pi*fn2*tbit)
            nosn = a1*np.sin(2*np.pi*fn*tbit)
        ask[i*len(tbit):(i+1)*len(tbit)] = zat
        psk[i*len(tbit):(i+1)*len(tbit)] = zpt
        fsk[i*len(tbit):(i+1)*len(tbit)] = zft
        nosny[i*len(tbit):(i+1)*len(tbit)] = nosn
    sygnaly = [ask, psk, fsk, nosny]
    return ask, psk, fsk, sygnaly

def koder74(dane):
    p1 = dane[0] ^ dane[1] ^ dane[3]
    p2 = dane[0] ^ dane[2] ^ dane[3]
    p3 = dane[1] ^ dane[2] ^ dane[3]
    zakodowaneDane = [p1, p2, dane[0], p3, dane[1], dane[2], dane[3]]
    return zakodowaneDane
def dekoder74(zakodowaneDane):
    p1 = int(zakodowaneDane[0])
    p2 = int(zakodowaneDane[1])
    p3 = int(zakodowaneDane[3])
    zakodowaneDane = [int(bit) for bit in zakodowaneDane]  # Konwersja na listę liczb całkowitych
    tab = [
        p1 ^ zakodowaneDane[2] ^ zakodowaneDane[4] ^ zakodowaneDane[6],
        p2 ^ zakodowaneDane[2] ^ zakodowaneDane[5] ^ zakodowaneDane[6],
        p3 ^ zakodowaneDane[4] ^ zakodowaneDane[5] ^ zakodowaneDane[6]
    ]
    error_bit = 0
    for i in range(3):
        error_bit += tab[i] * (2 ** i)
    if error_bit > 0:
        zakodowaneDane[error_bit - 1] ^= 1
    odkodowaneDane = [zakodowaneDane[2], zakodowaneDane[4], zakodowaneDane[5], zakodowaneDane[6]]
    return odkodowaneDane
def koder74_strumien(dane):
    strumienie = [list(map(int, dane[i:i+4])) for i in range(0, len(dane), 4)]
    zakodowane_strumienie = [koder74(strumien) for strumien in strumienie]
    return [bit for strumien in zakodowane_strumienie for bit in strumien]
def dekoder74_strumien(zakodowaneDane):
    strumienie = [zakodowaneDane[i:i+7] for i in range(0, len(zakodowaneDane), 7)]
    odkodowane_strumienie = [dekoder74(strumien) for strumien in strumienie]
    return [bit for strumien in odkodowane_strumienie for bit in strumien]

def ask_demodulacja(ask, sig_nosny):
    ct = np.zeros(len(ask) // int(tb * fs))
    ilosc_probek_na_bit = int(tb * fs)
    wartosci_pt = []
    for i in range(len(ask) // ilosc_probek_na_bit):
        xt = ask[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit] * sig_nosny[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit]
        pt = np.trapz(xt, dx=1/fs)
        wartosci_pt.append(pt)
    h=(np.max(wartosci_pt)+np.min(wartosci_pt))/2

    ct = wartosci_pt>h
    t = np.linspace(0, tb, int(tb * fs))
    return ct
def psk_demodulacja(psk, sig_nosny):
    ct = np.zeros(len(psk) // int(tb * fs))
    ilosc_probek_na_bit = int(tb * fs)
    wartosci_pt = []
    wartosci_xt = []    
    for i in range(len(psk) // ilosc_probek_na_bit):
        xt = psk[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit] * sig_nosny[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit]
        wartosci_xt.append(xt)
        pt = np.trapz(xt, dx=1/fs)
        wartosci_pt.append(pt)        
        if pt < 0:
            ct[i] = 1    
    t = np.linspace(0, tb, int(tb * fs))
    return ct
def fsk_demodulacja(fsk,A1,fn1,fn2):
    Tb=tc/B
    tbit=np.linspace(0,Tb,tc*fs)
    zft1=A1*np.sin(2*np.pi*fn1*tbit)
    zft2=A1*np.sin(2*np.pi*fn2*tbit)
    ct = np.zeros(len(fsk) // int(tb * fs))
    ilosc_probek_na_bit = int(tb * fs)
    pt1 = []
    pt2 = []
    pt3 = []
    xt1 = []
    xt2 = []    
    for i in range(len(fsk) // ilosc_probek_na_bit):
        xt_1 = fsk[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit] * zft1
        xt_2 = fsk[i * ilosc_probek_na_bit:(i + 1) * ilosc_probek_na_bit] * zft2
        xt1.append(xt_1)
        xt2.append(xt_2)
        pt_1 = np.trapz(xt_1, dx=1/fs)
        pt_2 = np.trapz(xt_2, dx=1/fs)
        pt_3 = (pt_1 * -1) + pt_2
        pt1.append(pt_1)
        pt2.append(pt_2)
        pt3.append(pt_3)        
        if pt_3 > 0:
            ct[i] = 1    
    t = np.linspace(0, tb, int(tb * fs))
    return ct    

def bit_error_rate(strumien, odebrane_bity):
    errors = sum(strumien[i] != odebrane_bity[i] for i in range(len(strumien)))
    ber = errors / len(strumien)
    return ber

def generuj_szum_bialy(szum_dlugosc, alfa):
    szum = np.random.uniform(-1, 1, szum_dlugosc)
    szum *= alfa
    return szum

def tlumienieSygnalu(beta,t):
    x= np.exp(-t * beta)
    return x


tc = 1
fs = 1000
fn = 10
tb = 1
strumien = '110010100111011010011001010011101101001100101001'
a1 = 1
a2 = 2
W = 2
alfa = 0.5
beta=10
t = np.linspace(0, 2, 1000)


print("Dane wejściowe:", strumien)

##zadanie1

#Zakodowanie strumienia kodem Hamminga
zakodowany_strumien = koder74_strumien(strumien)

B = len(zakodowany_strumien)
fn1 = (W+1)/(1/B)
fn2 = (W+2)/(1/B)

# Generowanie sygnału modulowanego
ask, psk, fsk, sygnaly = generujsygnal(zakodowany_strumien, tb, a1, a2, W, B)

# Demodulacja sygnału ASK
ct_ask = ask_demodulacja(ask, sygnaly[3]) #do zadania 1

# Dekodowanie strumienia kodem Hamminga
odkodowane_dane = dekoder74_strumien(ct_ask)
print("Odkodowane dane:", odkodowane_dane)

if strumien == ''.join(map(str, odkodowane_dane)):
    print("Strumienie bitowe są identyczne.")
else:
    print("Strumienie bitowe są różne.")

#zadanie2

#Zakodowanie strumienia kodem Hamminga
zakodowany_strumien = koder74_strumien(strumien)

B = len(zakodowany_strumien)
fn1 = (W+1)/(1/B)
fn2 = (W+2)/(1/B)

# Generowanie sygnału modulowanego
ask, psk, fsk, sygnaly = generujsygnal(zakodowany_strumien, tb, a1, a2, W, B)

szumm = generuj_szum_bialy(len(ask), alfa)
proba=ask+szumm

# Demodulacja sygnału ASK
ct_ask = ask_demodulacja(proba, sygnaly[3]) # do zadania 2

# Dekodowanie strumienia kodem Hamminga
odkodowane_dane = dekoder74_strumien(ct_ask)
print("Odkodowane dane:", odkodowane_dane)

if strumien == ''.join(map(str, odkodowane_dane)):
   print("Strumienie bitowe są identyczne.")
else:
   print("Strumienie bitowe są różne.")

alfas = np.linspace(0, 2, 11) 

ber_ask = []
ber_psk = []
ber_fsk = []

zakodowany_strumien = koder74_strumien(strumien)
odkodowany_zakodowany_strumien = dekoder74_strumien(zakodowany_strumien)

for alfa in alfas:
   B = len(zakodowany_strumien)
   ask, psk, fsk, sygnaly = generujsygnal(zakodowany_strumien, tb, a1, a2, W, B)

   szum = generuj_szum_bialy(len(ask), alfa)
   szum2 = generuj_szum_bialy(len(psk), alfa)
   szum3 = generuj_szum_bialy(len(fsk), alfa)

   proba_ask = ask + szum
   proba_psk = psk + szum2
   proba_fsk = fsk + szum3

   ct_ask = ask_demodulacja(proba_ask, sygnaly[3])
   ct_psk = psk_demodulacja(proba_psk, sygnaly[3])
   ct_fsk = fsk_demodulacja(proba_fsk, a1, fn1, fn2)

   odkodowane_dane_ask = dekoder74_strumien(ct_ask)
   odkodowane_dane_psk = dekoder74_strumien(ct_psk)
   odkodowane_dane_fsk = dekoder74_strumien(ct_fsk)

   ber_ask_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_ask)
   ber_psk_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_psk)
   ber_fsk_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_fsk)

   ber_ask.append(ber_ask_val)
   ber_psk.append(ber_psk_val)
   ber_fsk.append(ber_fsk_val)

    

plt.plot(alfas, ber_ask, label='ASK')
plt.plot(alfas, ber_psk, label='PSK')
plt.plot(alfas, ber_fsk, label='FSK')
plt.title('Zależność BER od parametru alfa')
plt.xlabel('Parametr alfa')
plt.ylabel('BER')
plt.legend()
plt.show()


##zadanie3

#Zakodowanie strumienia kodem Hamminga
zakodowany_strumien = koder74_strumien(strumien)

B = len(zakodowany_strumien)
fn1 = (W+1)/(1/B)
fn2 = (W+2)/(1/B)

# Generowanie sygnału modulowanego
ask, psk, fsk, sygnaly = generujsygnal(zakodowany_strumien, tb, a1, a2, W, B)

tlumienie=tlumienieSygnalu(beta,len(ask)) #do zadania3
proba2=ask*tlumienie

# Demodulacja sygnału ASK
ct_ask = ask_demodulacja(proba2, sygnaly[3]) # do zadania 3


# Dekodowanie strumienia kodem Hamminga
odkodowane_dane = dekoder74_strumien(ct_ask)
print("Odkodowane dane:", odkodowane_dane)

if strumien == ''.join(map(str, odkodowane_dane)):
   print("Strumienie bitowe są identyczne.")
else:
   print("Strumienie bitowe są różne.")


betas = np.linspace(0, 20, 11)

ber_ask = []
ber_psk = []
ber_fsk = []

zakodowany_strumien = koder74_strumien(strumien)
odkodowany_zakodowany_strumien = dekoder74_strumien(zakodowany_strumien)

for beta in betas:
   B = len(zakodowany_strumien)
   ask, psk, fsk, sygnaly = generujsygnal(zakodowany_strumien, tb, a1, a2, W, B)

   tlum=tlumienieSygnalu(beta,len(ask))
   tlum2=tlumienieSygnalu(beta,len(psk))
   tlum3=tlumienieSygnalu(beta,len(fsk))

   proba_ask = ask * tlum
   proba_psk = psk * tlum2
   proba_fsk = fsk * tlum3

   ct_ask = ask_demodulacja(proba_ask, sygnaly[3])
   ct_psk = psk_demodulacja(proba_psk, sygnaly[3])
   ct_fsk = fsk_demodulacja(proba_fsk, a1, fn1, fn2)

   odkodowane_dane_ask = dekoder74_strumien(ct_ask)
   odkodowane_dane_psk = dekoder74_strumien(ct_psk)
   odkodowane_dane_fsk = dekoder74_strumien(ct_fsk)

   ber_ask_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_ask)
   ber_psk_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_psk)
   ber_fsk_val = bit_error_rate(odkodowany_zakodowany_strumien, odkodowane_dane_fsk)

   ber_ask.append(ber_ask_val)
   ber_psk.append(ber_psk_val)
   ber_fsk.append(ber_fsk_val)

plt.plot(betas, ber_ask, label='ASK')
plt.plot(betas, ber_psk, label='PSK')
plt.plot(betas, ber_fsk, label='FSK')
plt.title('Zależność BER od parametru beta')
plt.xlabel('Parametr beta')
plt.ylabel('BER')
plt.legend()
plt.show()
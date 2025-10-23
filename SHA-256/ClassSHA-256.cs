// Класс для реализации стандартного алгоритма хэширования SHA-256
public class SHA256
{
    private static readonly uint[] K = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
       };

    // Основной метод для получения хэша
    public static string GetHash(string input)
    {
        // Инициализация начальных значений
        uint[] H = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        byte[] bytes = System.Text.Encoding.UTF8.GetBytes(input);

        // Предварительная обработка
        ulong bitLength = (ulong)bytes.Length * 8;
        bytes = bytes.Concat(new byte[] { 0x80 }).ToArray(); // Добавить 1
        while ((bytes.Length * 8) % 512 != 448)
        {
            bytes = bytes.Concat(new byte[] { 0x00 }).ToArray();
        }

        // Добавить длину сообщения
        byte[] length = BitConverter.GetBytes(bitLength);
        Array.Reverse(length);
        bytes = bytes.Concat(length).ToArray();

        // Обработка блоков
        for (int i = 0; i < bytes.Length; i += 64)
        {
            uint[] w = new uint[64];
            for (int t = 0; t < 16; t++)
            {
                int startIdx = i + t * 4;
                // Проверка индекса на выход за пределы массива
                if (startIdx < bytes.Length)
                {
                    w[t] = (uint)((bytes[startIdx] << 24) |
                                  (bytes[startIdx + 1] << 16) |
                                  (bytes[startIdx + 2] << 8) |
                                  bytes[startIdx + 3]);
                }
                else
                {
                    // Если мы достигли конца массива bytes, оставшиеся слова должны быть нулями
                    w[t] = 0;
                }
            }

            for (int t = 16; t < 64; t++)
            {
                uint s0 = RightRotate(w[t - 15], 7) ^ RightRotate(w[t - 15], 18) ^ (w[t - 15] >> 3);
                uint s1 = RightRotate(w[t - 2], 17) ^ RightRotate(w[t - 2], 19) ^ (w[t - 2] >> 10);
                w[t] = w[t - 16] + s0 + w[t - 7] + s1;
            }

            uint a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];

            for (int t = 0; t < 64; t++)
            {
                uint sum1 = RightRotate(e, 6) ^ RightRotate(e, 11) ^ RightRotate(e, 25);
                uint ch = (e & f) ^ (~e & g);
                uint temp1 = h + sum1 + ch + K[t] + w[t];
                uint sum0 = RightRotate(a, 2) ^ RightRotate(a, 13) ^ RightRotate(a, 22);
                uint maj = (a & b) ^ (a & c) ^ (b & c);
                uint temp2 = sum0 + maj;

                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            H[0] += a;
            H[1] += b;
            H[2] += c;
            H[3] += d;
            H[4] += e;
            H[5] += f;
            H[6] += g;
            H[7] += h;
        }

        // Формирование итогового хеша
        string hash = "";
        foreach (uint part in H)
        {
            hash += part.ToString("x8");
        }
        return hash;
    }

    // Метод для циклического сдвига битов вправо
    private static uint RightRotate(uint x, int n)
    {
        return (x >> n) | (x << (32 - n));
    }
}

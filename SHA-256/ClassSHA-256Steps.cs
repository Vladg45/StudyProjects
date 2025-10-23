// Класс для пошагового вычисления хэш-значения по алгоритму SHA-256 с регистрацией каждого шага
public class SHA256Steps
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

    // Основной метод для получения хэш-значения с регистрацией промежуточных шагов
    public static string GetHash(string input, List<string> steps)
    {
        // Инициализация начальных значений
        uint[] H = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        };

        // Преобразование входной строки в байты и регистрация процесса
        byte[] bytes = Encoding.UTF8.GetBytes(input);
        steps.Add($"Исходный текст: {input}");
        steps.Add($"Текст в байтах: {BitConverter.ToString(bytes)}");

        // Обработка входных байтов
        bytes = ProcessBytes(bytes, steps);

        // Окончательная обработка блоков
        uint[] finalHash = ProcessBlocks(bytes, H, steps);

        // Преобразование частей хеша в одну строку шестнадцатеричных чисел
        string hash = string.Concat(finalHash.Select(p => p.ToString("x8")));
        steps.Add($"Итоговый хеш: {hash}");
        return hash;
    }

    // Метод для предварительной обработки байтов входных данных
    private static byte[] ProcessBytes(byte[] input, List<string> steps)
    {
        // Добавление заполнения и длины в соответствии со спецификацией SHA-256
        ulong bitLength = (ulong)input.Length * 8;
        input = input.Concat(new byte[] { 0x80 }).ToArray();
        while ((input.Length * 8) % 512 != 448)
        {
            input = input.Concat(new byte[] { 0x00 }).ToArray();
        }

        byte[] lengthBytes = BitConverter.GetBytes(bitLength);
        Array.Reverse(lengthBytes);
        input = input.Concat(lengthBytes).ToArray();

        steps.Add($"Байты после добавления заполнения: {BitConverter.ToString(input)}");
        return input;
    }

    // Метод для обработки блоков данных и выполнения основных операций сжатия
    private static uint[] ProcessBlocks(byte[] bytes, uint[] H, List<string> steps)
    {
        // Обработка каждого блока байтов
        for (int i = 0; i < bytes.Length; i += 64)
        {
            uint[] w = new uint[64];
            steps.Add($"Обработка блока, начинающегося с байта индексом {i}");

            // Расширение сообщения в массив расписания сообщений w
            for (int t = 0; t < 16; t++)
            {
                int startIdx = i + t * 4;
                w[t] = (uint)((bytes[startIdx] << 24) | (bytes[startIdx + 1] << 16) | (bytes[startIdx + 2] << 8) | bytes[startIdx + 3]);

                steps.Add($"Значения W на {t + 1} шаге: {string.Join(", ", w.Select(k => k.ToString("x")))}");
            }

            // Расширение первых 16 слов в оставшиеся 64 слова
            for (int t = 16; t < 64; t++)
            {
                uint s0 = RightRotate(w[t - 15], 7) ^ RightRotate(w[t - 15], 18) ^ (w[t - 15] >> 3);
                uint s1 = RightRotate(w[t - 2], 17) ^ RightRotate(w[t - 2], 19) ^ (w[t - 2] >> 10);
                w[t] = w[t - 16] + s0 + w[t - 7] + s1;

                steps.Add($"Значения W на {t + 1} шаге: {string.Join(", ", w.Select(k => k.ToString("x")))}");
            }

            // Инициализация рабочих переменных
            uint a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];

            // Основной цикл функции сжатия
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

                steps.Add($"Значения H на {t + 1} шаге: {a}, {b}, {c}, {d}, {e}, {f}, {g}, {h}");
            }

            // Добавление сжатого блока к текущему значению хеша
            H[0] += a;
            H[1] += b;
            H[2] += c;
            H[3] += d;
            H[4] += e;
            H[5] += f;
            H[6] += g;
            H[7] += h;
        }

        steps.Add($"Промежуточные значения хеша: {string.Join(", ", H.Select(x => x.ToString("x8")))}");
        return H;
    }

    // Метод для циклического сдвига битов вправо    
    private static uint RightRotate(uint x, int n)
    {
        return (x >> n) | (x << (32 - n));
    }
}

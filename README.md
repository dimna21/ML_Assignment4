# FER2013 emotion recognition
ექსპერიმენტი მიზნად ისახავს სურათების დამუშავების სხვადასხვა ტექნიკების გატესტვას და მათი საუკეთესო კომბინაციის ამორჩევას FER2013 დატასეტზე ემოციების პრედიქციისთვის.

# Data preprocessing
დატასეტში არის ემოციების 7 კლასი: class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]. თითოეული მონაცემი არის 48x48 პიქსელიანი greyscale სურათი.
პრეპროცესინგის ნაწილში პიქსელების მნიშვნელობები სკალირდება 0-1 რეინჯში, რათა თავი დავიზღვიოთ exploding gradient პრობლემისგან. დატასეტს აქვს იმბალანსის პრობლემაც, ამიტომ 
ზოგიერთ ექსპერიმენტში გატესტილია upsampling მეთოდი, რომლის დროსაც უმცირესობაში მყოფი კლასების სურათებს +-10 intensity ემატება და მონაცემთა ბაზაში მცირედი სახეცვლილებით  დუბლირებულად ბრუნდება.
თეორიულ დონეზე, პიქსელების ინტენსივობაში მცირედი shift გამოსადეგი უნდა იყოს მოდელის განზოგადების უნარისთვის. შესაძლებელი იყო rotating, mirroring და სხვა ტექნიკების გამოყენებაც, მაგრამ ტესტ-სეტში მსგავსი
გადახრები არ გვხვდება, ამიტომ ეს ტრანსფორმაციები მოდელის პერფორმანსს ვერ გააუმჯობესებენ.

# Neural network iterations
1. პირველი მოდელი (BaselineModel კლასი) მიზნად ისახავდა ზოგადი წარმოდგენის შექმნას მარტივი არქიტექტურის პერფორმანსზე. არქიტექტურა შემდეგია:
32 filter conv2d -> maxpooling(2,2) -> ReLU -> 64 filter conv2d -> maxpooling(2,2) -> ReLU -> 128 filter conv2d -> maxpooling(2,2) -> ReLU -> Flatten -> Linear(128) -> Output(7). ამ ექსპერიმენტის შედეგად დაიდო 0.77/0.51 train/validation accuracy. იგივე მოდელი გავუშვი მეორედ, ამჯერად დატასეტის დაბალანსების ტექნიკით და მოდელმა მცირედ გაუმჯობესებას მიაღწია: 0.67/0.54 accuracy. პირველადი მოდელის ორმა ექსპერიმენტმა გვანახა, რომ ეს არქიტექტურა გადის overfitში და დატასეტის დაბალანსების ტექნიკას გარკვეული სარგებელი მოაქვს.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/7amufnzs)
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/dnyp6qje)

2. მოდელის შემდგომი იტერაცია (ImprovedModel კლასი ყოველ იტერაციაზე იცვლება) BaselineModel-ს ამატებს Batchnorm ლეიერებს კონვოლუციურ შრეებსა და head-ში. ამის ხარჯზე მოდელი 0.70/0.58 accuracy-ზე გადის, რადგან გრადიენტი ნორმალიზაციის დახმარებით ნეირონულ ქსელში უკეთესი მაგნიტუდით ვრცელდება.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/cuxqkkdm)

3. მოდელს მეორე გაუმჯობესებად კონვოლუციურ შრეებსა და head-ში ემატება dropout ლეიერები overfit-ის შესამცირებლად, რისი მეშვეობითაც 0.62/0.60 accuracyზე გადის. (https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/ydlxqyzi)

4. ამ ეტაპზე მოდელი overfit-ში აღარ გადის, მაგრამ მისი უნარიანობის გასაუმჯობესებლად გავზარდე კომპლექსურობა: დავამატე კიდევ ერთი 256 ფილტრიანი კონვოლუციური ბლოკი, რომელმაც პერფორმანსი ვერ გაზარდა. (https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/lqfkix8t)

5. მოდელის გაუმჯობესების შემდეგი მცდელობა იყო უფრო კომპლექსური head ბლოკის გაკეთება 512->256->128->7 ფორმით, რამაც accuracy 0.64/0.61-მდე გაზარდა.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/65qgarhf)

6. ზემოხსენებული კომპონენტების კომპლექსურობის კიდევ უფრო ზრდას შედეგი არ გამოუღია და საჭირო გახდა არქიტექტურული ცვლილება, რისთვისაც მოდელს დავამატე residual ბლოკები. მისი არქიტექტურა აღწერილია GigaModel კლასში. Residual კავშირებმა შედეგი ვერ გააუმჯობესა.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/cyn9apy4)

7. ამ იტერაციაზე წინა მოდელს კიდევ 2 კონვოლუციური და ერთი fc შრე დავამატე, თუმცა ვალიდაციის შედეგი 60%-ს მაინც არ აცდა.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/ofsajdfh)

8. საბოლოოდ ვცადე მონაცემთა პრეპროცესინგის მიდგომის ცვლილება: ჩავხედე confusion matrix-ს და უფრო აგრესიული აფსემპლინგი გავუკეთე იმ კლასებს, რომლებზეც მოდელს სუსტი პერფორმანსი ჰქონდა, მაგრამ ამ მიდგომით მოდელი 0.61/0.58 accuracy-ზე გაჩერდა.
(https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/fufcp077) (https://wandb.ai/dimna21-free-university-of-tbilisi-/ML_Assignment4/runs/gltl4ney)
